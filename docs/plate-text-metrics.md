# Strict paired plate text evaluation

Python standard library only. No OCR model, downloads, training or images needed. This is a reusable measurement tool, not a recognition improvement.

    python scripts/plate_text_metrics.py --truth TRUTH.json --predictions PREDICTIONS.json --output NEW_RESULT.json
    python -m pytest -q tests/test_plate_text_metrics.py

Each input is a nonempty JSON list of rows with exactly three string fields: id, letters_rtl, digits_ltr. A row could use id fictional-1, letters_rtl أبس and digits_ltr ١٢٣. No real plate inputs are bundled. Letters are in logical right-to-left reading order as a string, not reverse visual coordinates. Digits are in visual left-to-right sequence. State that convention before collecting labels. Ground truth needs nonempty letters and digits. Prediction failures must be explicit empty strings, not missing rows. IDs must match exactly, so a caller cannot silently exclude unsuccessful plates.

Metrics: whole-plate exact match, separate letter/digit edit counts and rates, and groupwise character edit rate, total separate-group edits divided by total truth characters. This is not mixed-script full-string CER. Insertions can make error rate exceed1. No normalization of hamza, dotlessyeh, digit script, ordering or whitespace. Such changes must never turn errors into successes silently. Arabic versus Latin numerals differ. Do not confuse accuracy and1minusCER when insertions occur.

Outputs omit raw registration text, but IDs can still identify records. Keep files private and choose nonidentifying IDs. No permission to process, distribute or publish a real plate dataset follows from this code. All bundled sample registrations are fictional strings, not looked up and could coincide with actual registrations by chance.

Twelve unit tests pass: edit cases, exact matches, blank predictions, insertion behavior, missing/extra IDs, ordering, exactUnicode, no raw-text output, input validation and no-overwriteCLI. Additional1,000fixed-seed random edit-distance pairs matched a separate full-matrix reference implementation. Fictional sample result:1/2exact,6/12characters edited,1empty prediction. This is a wiring example, not real accuracy.

The tool does not validate text against pixels or official plate assignment rules. It does not enforce a letter alphabet, because historical/foreign/annotation formats should not be silently rewritten. It accepts letters and decimal digits only, maximum32pergroup. It does not group multiple plates, distinguish localization versus recognition failures, estimate confidence intervals or certify vehicle separation. Current real data is exposed and insufficient for independent end-to-end claims.

This recipe is staged for review, not merged to main. No fees or paid compute. New code MIT, datasets/models/third-party rights remain separate.
