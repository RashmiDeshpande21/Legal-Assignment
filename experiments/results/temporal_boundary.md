# As-of boundary stress test

**PASS** — 2223 lattice assertions over 57 amended nodes × 13 dates, 0 failures.

The 12-question set only asks for 2018-01-01 (before First) and 2020-06-01 (after Second, still before Third). This test covers every amendment boundary the graph actually has — day-before / day-of / day-after for First, Second, and Third — plus two mid-window sentinels.

## Dates tested

`2017-12-31`, `2018-06-25`, `2018-06-26`, `2018-06-27`, `2019-01-01`, `2020-05-12`, `2020-05-13`, `2020-05-14`, `2020-08-01`, `2020-12-14`, `2020-12-15`, `2020-12-16`, `2021-01-01`

## Window `2019-01-01` — between First and Second

4 nodes must read as amended and 53 must still read as originally executed (future instruments must not leak into citations).

| Node | Instruments | In force at window | Status | Citation |
|---|---|---|---|---|
| §1.03(b) | First Amendment | First Amendment | amended | Credit Agreement §1.03(b), as amended by First Amendment (effective 2018-06-26) |
| §7.04(e) | First Amendment, Third Amendment | First Amendment | amended | Credit Agreement §7.04(e), as amended by First Amendment (effective 2018-06-26) |
| definition of “Consolidated EBITDA” (§1.01) | First Amendment | First Amendment | amended | Credit Agreement definition of “Consolidated EBITDA” (§1.01), as amended by First Amendment (effective 2018-06-26) |
| Exhibit C | First Amendment | First Amendment | amended | Credit Agreement Exhibit C, as amended by First Amendment (effective 2018-06-26) |

(53 not-yet-effective nodes verified as still `original` at this window.)

## Window `2020-08-01` — between Second and Third

34 nodes must read as amended and 23 must still read as originally executed (future instruments must not leak into citations).

| Node | Instruments | In force at window | Status | Citation |
|---|---|---|---|---|
| definition of “Affected Financial Institution” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “Affected Financial Institution” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Consolidated Cash on Hand” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “Consolidated Cash on Hand” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “LIBOR” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “LIBOR” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Liquidity” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “Liquidity” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Resolution Authority” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “Resolution Authority” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Second Amendment Effective Date” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “Second Amendment Effective Date” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “UK Financial Institution” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “UK Financial Institution” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “UK Resolution Authority” (§1.01) | Second Amendment | Second Amendment | amended | Second Amendment to Third Amended and Restated Credit Agreement definition of “UK Resolution Authority” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| §1.01 | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement §1.01, as amended by Second Amendment (effective 2020-05-13) |
| §1.03(b) | First Amendment | First Amendment | amended | Credit Agreement §1.03(b), as amended by First Amendment (effective 2018-06-26) |
| §1.08 | Second Amendment | Second Amendment | added | Credit Agreement §1.08, as amended by Second Amendment (effective 2020-05-13) |
| §10.22 | Second Amendment | Second Amendment | amended | Credit Agreement §10.22, as amended by Second Amendment (effective 2020-05-13) |
| §10.23 | Second Amendment | Second Amendment | added | Credit Agreement §10.23, as amended by Second Amendment (effective 2020-05-13) |
| §2.15(a)(iv) | Second Amendment | Second Amendment | amended | Credit Agreement §2.15(a)(iv), as amended by Second Amendment (effective 2020-05-13) |
| §5.05(c) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement §5.05(c), as amended by Second Amendment (effective 2020-05-13) |
| §5.23 | Second Amendment | Second Amendment | amended | Credit Agreement §5.23, as amended by Second Amendment (effective 2020-05-13) |
| §6.01 | Second Amendment | Second Amendment | amended | Credit Agreement §6.01, as amended by Second Amendment (effective 2020-05-13) |
| §7.03 | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement §7.03, as amended by Second Amendment (effective 2020-05-13) |
| §7.04(e) | First Amendment, Third Amendment | First Amendment | amended | Credit Agreement §7.04(e), as amended by First Amendment (effective 2018-06-26) |
| §7.05(a) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement §7.05(a), as amended by Second Amendment (effective 2020-05-13) |
| §7.10 | Second Amendment | Second Amendment | amended | Credit Agreement §7.10, as amended by Second Amendment (effective 2020-05-13) |
| §7.10(a) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement §7.10(a), as amended by Second Amendment (effective 2020-05-13) |
| §7.10(b) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement §7.10(b), as amended by Second Amendment (effective 2020-05-13) |
| §7.19 | Second Amendment, Third Amendment | Second Amendment | added | Credit Agreement §7.19, as amended by Second Amendment (effective 2020-05-13) |
| definition of “Applicable Rate” (§1.01) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement definition of “Applicable Rate” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Bail-In Action” (§1.01) | Second Amendment | Second Amendment | amended | Credit Agreement definition of “Bail-In Action” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Bail-In Legislation” (§1.01) | Second Amendment | Second Amendment | amended | Credit Agreement definition of “Bail-In Legislation” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Base Rate” (§1.01) | Second Amendment | Second Amendment | amended | Credit Agreement definition of “Base Rate” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Consolidated EBITDA” (§1.01) | First Amendment | First Amendment | amended | Credit Agreement definition of “Consolidated EBITDA” (§1.01), as amended by First Amendment (effective 2018-06-26) |
| definition of “Consolidated Leverage Ratio” (§1.01) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement definition of “Consolidated Leverage Ratio” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Eurodollar Rate” (§1.01) | Second Amendment | Second Amendment | amended | Credit Agreement definition of “Eurodollar Rate” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Eurodollar Rate Loan” (§1.01) | Second Amendment, Third Amendment | Second Amendment | amended | Credit Agreement definition of “Eurodollar Rate Loan” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| definition of “Write-Down and Conversion Powers” (§1.01) | Second Amendment | Second Amendment | amended | Credit Agreement definition of “Write-Down and Conversion Powers” (§1.01), as amended by Second Amendment (effective 2020-05-13) |
| Exhibit C | First Amendment | First Amendment | amended | Credit Agreement Exhibit C, as amended by First Amendment (effective 2018-06-26) |

(23 not-yet-effective nodes verified as still `original` at this window.)

## Mixed vintage (2019-01-01)

2 of 2 dependency chains are mixed-vintage: the definition a provision relies on is amended while the provision's own text is not.

| Amended input | Dependent | Dependent at mid | Dependent after all |
|---|---|---|---|
| base::def::Consolidated EBITDA (amended) | base::def::Consolidated EBITDAR | original | original |
| base::def::Consolidated EBITDA (amended) | base::def::Consolidated Leverage Ratio | original | amended |
