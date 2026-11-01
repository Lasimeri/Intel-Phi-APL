# shape_check.awk: hold what the compiler inferred against what ran (TEST ONLY).
#   awk -f shape_check.awk INFERRED RUNLOG
# INFERRED: "S id d0 d1 ..." with ¯1 = T. RUNLOG: "P p" at each step, "S id d0 d1 ...".
# Every line run must match its row, T being exactly p+1. Exit 0 pass, 1 fail.
FNR == NR {
    if ($1 == "S") { id = $2; $1 = ""; $2 = ""; want[id] = $0; gsub(/^ +/, "", want[id]); nrow++ }
    next
}
$1 == "P" { p = $2 + 0; next }
$1 != "S" { next }
{
    id = $2; $1 = ""; $2 = ""; got = $0; gsub(/^ +/, "", got); checked++
    if (!(id in want)) { print "no inference for row " id; bad++; next }
    seen[id] = 1
    w = want[id]; gsub(/¯1/, p + 1, w)
    if (w != got) { if (bad < 12) print "row " id ": inferred [" want[id] "] ran [" got "] at p=" p; bad++ }
}
END {
    for (id in want) if (!(id in seen)) unrun++
    printf "shapes: %d rows inferred, %d lines checked, %d mismatches, %d rows never ran\n", nrow, checked, bad + 0, unrun + 0
    exit bad > 0 || unrun > 0
}
