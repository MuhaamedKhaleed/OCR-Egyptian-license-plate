# 500 Egyptian-style synthetic plates, 25 November 2026

500 unique fictional registrations with exact labels. 400 train, 50 development, 50 test. This pilot covers only Cairo-style three letters plus three digits, not all Egyptian plate formats or actual registration assignment. Digits1–9. Letters: أ ب ج د ر س ص ط ع ف ق ل م ن ه و ي. ي is rendered in the dotless plate-style form; ه uses an isolated printed form. Strings are invented and may coincide with real registrations by chance, no registration lookup was performed.

Programmatic image rendering, not photos of real vehicles and not an image-generation model. Blue/white plate-like layout, three OFL-licensed Noto Arabic font families, lighting/noise/blur variation. Generic font shapes are visibly different from embossed real plates. No security seals or authentic manufacturing assets. Do not use as a document or actual plate.

## Actual training

Small CNN trained from scratch for25epochs on2400glyphs cropped using known renderer boxes. Seed42. Best checkpoint chosen by dev character accuracy. Synthetic test:300of300characters and50of50whole strings correct. This is oracle-segmented synthetic performance, NOT end-to-end OCR and NOT real-plate accuracy. Font/layout families are shared across splits. Exact strings and image bytes are unique across the splits, but that does not make this an external-domain test.

A tiny exploratory real-glyph check gave only3correct of21 manually boxed characters on three selected clear plate crops. It has manually read labels, selected boxes and exposed-domain source images previously used for localizer training. It is not a representative benchmark or independent test. It is enough to show the current synthetic setup does not transfer well, even with character locations supplied. No score increase or claim that OCR is solved.

PNG-source training was repeated and reproduced exactly. The shipped JPEG dataset was trained separately and gives the recorded report, not assumed equal from image conversion.

## Run

Python3.11. Numpy1.26.4, Pillow11.1.0, torch2.4.1+cpu. These versions were available in the local CPU environment. Default font paths are Linux system paths. For other systems, pass one or more --font PATH options for local Arabic fonts plus --latin-font PATH. Missing fonts fail before creating output. Install the OFL-licensed Noto Arabic fonts locally before regenerating. No automatic downloads, remote code or credentials.

Generate into a fresh directory:

    python scripts/synthetic_plate_cpu.py generate --data NEW_DATA
    python scripts/synthetic_plate_cpu.py train --data NEW_DATA --output NEW_MODEL

All outputs must be new. Before training, the recipe validates all500images, split counts, alphabet, box bounds, paths, duplicate registrations and identical image hashes. This is structural validation, not evidence of realistic appearance. The privately delivered best.pt is a local state-dict artifact, load only with weights_only=True using the architecture in the script. No inference confidence calibration or full-plate decoder is claimed. The real diagnostic needs private local source crops, which are not bundled. No real diagnostic script or transcripts are published.

## Sources and rights

Egyptian letter exclusions and1–9 digits: Egyptian reporting quoting a former traffic director, https://lite.almasryalyoum.com/extra/133630/ . Egyptian alphabet table and Cairo format: https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Egypt . Cross-checked against Arabic Egyptian table. The public auction site's FAQ confirms restricted letters but its composer was not accessible without sign-in: https://lo7tak.com.eg/Lo7tak/f/UserPages/FAQ.xhtml . This is not an official legal certification of every possible registration.

Noto fonts are SIL Open Font License1.1. Noto OFL terms: https://openfontlicense.org/open-font-license-official-text/ . No font binaries redistributed. No font binaries included. The font-output license is separate from code. Real diagnostic source: https://www.kaggle.com/datasets/mahmoudkhater99/car-plate , self-collected source with DbCL/ODbL as reviewed in the previous localization experiment. No real source images or plate transcripts are shipped here.

Code is offered under MIT, with third-party font and data rights separate. No fees or main merge. Only code and documentation enter the review branch, not images or weights. The project portfolio score stays60.
