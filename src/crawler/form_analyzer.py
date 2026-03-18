"""Form analysis — identifies form fields, types, and validation."""

from __future__ import annotations

import hashlib
import logging

from playwright.async_api import Page

from src.models.site_model import FormField, FormModel

logger = logging.getLogger(__name__)


async def analyze_forms(page: Page) -> list[FormModel]:
    """Analyze all forms on a page and return structured form models."""
    try:
        raw_forms = await page.evaluate("""() => {
            const forms = document.querySelectorAll('form');
            return Array.from(forms).map((form, fi) => {
                const fields = [];
                const inputs = form.querySelectorAll('input, select, textarea');

                for (const inp of inputs) {
                    const tag = inp.tagName.toLowerCase();
                    let fieldType = 'text';
                    let options = null;

                    if (tag === 'select') {
                        fieldType = 'select';
                        options = Array.from(inp.options).map(o => o.value).filter(v => v);
                    } else if (tag === 'textarea') {
                        fieldType = 'textarea';
                    } else if (tag === 'input') {
                        fieldType = inp.type || 'text';
                    }

                    // Skip hidden/submit fields
                    if (['hidden', 'submit', 'button', 'reset', 'image'].includes(fieldType)) continue;

                    let selector = '';
                    if (inp.id) selector = `#${CSS.escape(inp.id)}`;
                    else if (inp.name) selector = `${tag}[name="${inp.name}"]`;
                    else selector = `form:nth-of-type(${fi + 1}) ${tag}:nth-of-type(${Array.from(form.querySelectorAll(tag)).indexOf(inp) + 1})`;

                    fields.push({
                        name: inp.name || inp.id || '',
                        field_type: fieldType,
                        required: inp.required || inp.getAttribute('aria-required') === 'true',
                        validation_pattern: inp.pattern || null,
                        options: options,
                        selector: selector,
                    });
                }

                // Find submit button (must be visible)
                let submitSelector = '';
                const isVisible = (el) => {
                    if (!el) return false;
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    return rect.width > 0 && rect.height > 0
                        && style.visibility !== 'hidden'
                        && style.display !== 'none'
                        && !el.classList.contains('hidden');
                };
                const submitBtns = Array.from(form.querySelectorAll('button[type="submit"], input[type="submit"]'));
                const submitBtn = submitBtns.find(isVisible);
                if (submitBtn) {
                    if (submitBtn.id) submitSelector = `#${CSS.escape(submitBtn.id)}`;
                    else submitSelector = `form:nth-of-type(${fi + 1}) button[type="submit"], form:nth-of-type(${fi + 1}) input[type="submit"]`;
                } else {
                    const anyBtns = Array.from(form.querySelectorAll('button, [role="button"]'));
                    const anyBtn = anyBtns.find(isVisible);
                    if (anyBtn) {
                        if (anyBtn.id) submitSelector = `#${CSS.escape(anyBtn.id)}`;
                        else submitSelector = `form:nth-of-type(${fi + 1}) button`;
                    }
                }

                return {
                    action: form.action || '',
                    method: (form.method || 'GET').toUpperCase(),
                    fields: fields,
                    submit_selector: submitSelector,
                };
            });
        }""")

        forms = []
        for i, raw in enumerate(raw_forms):
            fid = hashlib.md5(f"form:{i}:{raw.get('action', '')}".encode()).hexdigest()[:10]
            fields = [FormField(**f) for f in raw.get("fields", [])]
            forms.append(
                FormModel(
                    form_id=fid,
                    action=raw.get("action", ""),
                    method=raw.get("method", "GET"),
                    fields=fields,
                    submit_selector=raw.get("submit_selector", ""),
                )
            )

        # Enrich with Angular Material component detection (parallel).
        # These are read-only page.evaluate() calls — safe to run concurrently.
        import asyncio

        mat_tasks = [
            _detect_angular_material_fields(page, fm) for fm in forms
        ]
        orphan_task = asyncio.create_task(_detect_angular_material_forms(page))
        mat_tasks_gathered = asyncio.gather(*mat_tasks, return_exceptions=True)

        await mat_tasks_gathered  # enrichment is in-place on form_model
        try:
            angular_forms = await orphan_task
            forms.extend(angular_forms)
        except Exception:
            pass

        logger.debug("Analyzed %d forms", len(forms))
        return forms

    except Exception as e:
        logger.error("Form analysis failed: %s", e)
        return []


async def _detect_angular_material_fields(page: Page, form_model: FormModel) -> None:
    """Enrich form fields with Angular Material interaction patterns."""
    try:
        enrichments = await page.evaluate("""(submitSel) => {
            const results = [];
            // Find the form element by its submit button
            let formEl = null;
            if (submitSel) {
                const btn = document.querySelector(submitSel);
                if (btn) formEl = btn.closest('form') || btn.closest('mat-card') || document.body;
            }
            if (!formEl) formEl = document.body;

            // Detect mat-select fields
            formEl.querySelectorAll('mat-select').forEach(sel => {
                const ff = sel.closest('mat-form-field');
                const label = ff?.querySelector('mat-label')?.textContent?.trim() || '';
                const id = sel.id || '';
                const triggerSel = id ? `#${CSS.escape(id)}` : 'mat-select';

                results.push({
                    name: label || id,
                    selector: triggerSel,
                    field_type: 'select',
                    interaction_pattern: 'mat_select',
                    interaction_steps: [
                        {action_type: 'click', selector: triggerSel, description: `Open ${label} dropdown`},
                        {action_type: 'click_text', selector: '', description: `Select option from ${label}`},
                    ],
                    required: sel.hasAttribute('required') || sel.getAttribute('aria-required') === 'true',
                    options: [],  // Can't read without opening
                });
            });

            // Detect mat-datepicker fields
            formEl.querySelectorAll('input[matDatepicker], input[matdatepicker]').forEach(inp => {
                const ff = inp.closest('mat-form-field');
                const label = ff?.querySelector('mat-label')?.textContent?.trim() || '';
                const id = inp.id || '';
                const sel = id ? `#${CSS.escape(id)}` : 'input[matDatepicker]';

                results.push({
                    name: label || id,
                    selector: sel,
                    field_type: 'date',
                    interaction_pattern: 'mat_datepicker',
                    interaction_steps: [
                        {action_type: 'fill', selector: sel, description: `Type date in ${label}`},
                    ],
                    required: inp.hasAttribute('required'),
                    options: null,
                });
            });

            // Detect mat-checkbox fields
            formEl.querySelectorAll('mat-checkbox').forEach(cb => {
                const label = cb.textContent?.trim()?.substring(0, 50) || '';
                const id = cb.id || cb.querySelector('input')?.id || '';
                const sel = id ? `#${CSS.escape(id)}` : `mat-checkbox`;

                results.push({
                    name: label,
                    selector: sel,
                    field_type: 'checkbox',
                    interaction_pattern: 'mat_checkbox',
                    interaction_steps: [
                        {action_type: 'click', selector: sel, description: `Toggle ${label}`},
                    ],
                    required: false,
                    options: null,
                });
            });

            // Detect mat-slide-toggle fields
            formEl.querySelectorAll('mat-slide-toggle').forEach(tog => {
                const label = tog.textContent?.trim()?.substring(0, 50) || '';
                const id = tog.id || '';
                const sel = id ? `#${CSS.escape(id)}` : 'mat-slide-toggle';

                results.push({
                    name: label,
                    selector: sel,
                    field_type: 'toggle',
                    interaction_pattern: 'mat_slide_toggle',
                    interaction_steps: [
                        {action_type: 'click', selector: sel, description: `Toggle ${label}`},
                    ],
                    required: false,
                    options: null,
                });
            });

            // Detect validation rules from mat-error elements
            formEl.querySelectorAll('mat-error, .mat-mdc-form-field-error').forEach(err => {
                const ff = err.closest('mat-form-field');
                if (!ff) return;
                const inp = ff.querySelector('input, textarea, mat-select');
                const id = inp?.id || '';
                if (id) {
                    results.push({
                        _validation_for: id,
                        error_message: err.textContent?.trim()?.substring(0, 100) || '',
                    });
                }
            });

            // Detect wizard (mat-stepper)
            const stepper = formEl.querySelector('mat-stepper, mat-horizontal-stepper, mat-vertical-stepper');
            let wizardSteps = [];
            if (stepper) {
                stepper.querySelectorAll('mat-step').forEach((step, idx) => {
                    const header = stepper.querySelectorAll('mat-step-header')[idx];
                    const label = header?.textContent?.trim()?.substring(0, 50) || `Step ${idx + 1}`;
                    wizardSteps.push({ step_index: idx, label });
                });
            }

            return { fields: results, wizard_steps: wizardSteps };
        }""", form_model.submit_selector)

        if not enrichments or not isinstance(enrichments, dict):
            return

        # Add Angular Material fields that aren't already in the form
        existing_selectors = {f.selector for f in form_model.fields}
        for field_data in enrichments.get("fields", []):
            if "_validation_for" in field_data:
                # Attach validation rule to existing field
                target_id = field_data["_validation_for"]
                for f in form_model.fields:
                    if target_id in f.selector:
                        f.validation_rules.append({
                            "rule": "error_message",
                            "value": field_data["error_message"],
                        })
                continue

            sel = field_data.get("selector", "")
            if sel in existing_selectors:
                # Enrich existing field
                for f in form_model.fields:
                    if f.selector == sel:
                        f.interaction_pattern = field_data.get("interaction_pattern", "standard")
                        f.interaction_steps = field_data.get("interaction_steps", [])
                        break
            else:
                # New Angular Material field
                form_model.fields.append(FormField(
                    name=field_data.get("name", ""),
                    field_type=field_data.get("field_type", "text"),
                    required=field_data.get("required", False),
                    options=field_data.get("options"),
                    selector=sel,
                    interaction_pattern=field_data.get("interaction_pattern", "standard"),
                    interaction_steps=field_data.get("interaction_steps", []),
                ))

        # Set wizard pattern
        wizard_steps = enrichments.get("wizard_steps", [])
        if wizard_steps:
            form_model.form_pattern = "wizard"
            form_model.wizard_steps = wizard_steps

    except Exception as e:
        logger.debug("Angular Material field detection failed: %s", e)


async def _detect_angular_material_forms(page: Page) -> list[FormModel]:
    """Detect Angular Material form components that don't use <form> wrappers."""
    try:
        raw = await page.evaluate("""() => {
            const results = [];
            // Find mat-form-fields not inside a <form>
            const orphanFields = document.querySelectorAll('mat-form-field');
            const groupedByCard = new Map();

            for (const ff of orphanFields) {
                if (ff.closest('form')) continue;  // Already handled
                const container = ff.closest('mat-card, mat-dialog-container, .mat-mdc-card, [class*="panel"]') || document.body;
                const key = container.id || container.className?.split(' ')[0] || 'orphan';
                if (!groupedByCard.has(key)) groupedByCard.set(key, []);
                groupedByCard.get(key).push(ff);
            }

            for (const [key, fields] of groupedByCard) {
                if (fields.length === 0) continue;
                const formFields = [];

                for (const ff of fields) {
                    const inp = ff.querySelector('input, textarea, mat-select');
                    if (!inp) continue;
                    const label = ff.querySelector('mat-label')?.textContent?.trim() || '';
                    const id = inp.id || '';
                    const sel = id ? `#${CSS.escape(id)}` : '';
                    if (!sel) continue;

                    const isMaterialSelect = inp.tagName.toLowerCase() === 'mat-select';
                    formFields.push({
                        name: label || id,
                        field_type: isMaterialSelect ? 'select' : (inp.type || 'text'),
                        required: inp.hasAttribute('required') || inp.getAttribute('aria-required') === 'true',
                        selector: sel,
                        interaction_pattern: isMaterialSelect ? 'mat_select' : 'standard',
                        interaction_steps: isMaterialSelect ? [
                            {action_type: 'click', selector: sel, description: `Open ${label}`},
                            {action_type: 'click_text', selector: '', description: `Select from ${label}`},
                        ] : [],
                    });
                }

                if (formFields.length > 0) {
                    // Check if the container is a dialog
                    const isDialog = !!fields[0].closest('mat-dialog-container, [role="dialog"]');
                    results.push({
                        fields: formFields,
                        form_pattern: isDialog ? 'dialog' : 'standard',
                    });
                }
            }

            return results;
        }""")

        if not raw or not isinstance(raw, list):
            return []

        # Filter: orphan forms must have form_pattern (set by our JS, not present in standard forms)
        raw = [r for r in raw if isinstance(r, dict) and "form_pattern" in r and "fields" in r]

        forms = []
        for i, r in enumerate(raw):
            fid = hashlib.md5(f"mat_form:{i}".encode()).hexdigest()[:10]
            fields = [FormField(**f) for f in r.get("fields", [])]
            forms.append(FormModel(
                form_id=fid,
                fields=fields,
                form_pattern=r.get("form_pattern", "standard"),
            ))

        if forms:
            logger.debug("Detected %d Angular Material form groups", len(forms))
        return forms

    except Exception as e:
        logger.debug("Angular Material form detection failed: %s", e)
        return []
