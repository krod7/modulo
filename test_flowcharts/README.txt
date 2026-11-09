MODULO TEST FLOWCHARTS
======================
Folders: svg/ png/ jpg/ jpeg/ pdf/  -> the same 15 valid charts in every format (single page each)
         invalid/   -> 4 broken charts (same 5 formats) for testing the validation service
         stress/    -> low-res, 2-degree rotated, and noisy/blurred versions of 3 charts
manifest.csv lists each chart and the constructs it tests.
gen_flowcharts.py regenerates everything (pip install cairosvg pillow numpy).

Conventions: rounded box = Start/End, rectangle = process, diamond = decision (Yes/No labels),
parallelogram = Input/Output. Black on white, ASCII-only text, no title text on the page.
