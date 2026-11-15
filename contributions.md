# Individual Contribution Statement & Git Log

**Project**: P2 — Home Energy Planner  
**Author**: Yousuf Rehab  
**Date**: November 2026

---

## 1. Individual Contribution Statement

> **Yousuf Rehab** (155 words):  
> I developed the Home Energy Planner project end-to-end, covering the data engineering, neural forecasting, constraint satisfaction, and graph search components. I built the data pipeline to clean and resample the UCI household power dataset into hourly intervals, filled missing gaps, and enforced a strict chronological split. I then designed and trained the PyTorch neural network to forecast 24-hour demand, validating that it beat both naive persistence baselines. For appliance scheduling, I implemented the CSP solver supporting Backtracking, Forward Checking, and AC-3 to enforce device precedence, shared circuit limits, and power caps. For the battery system, I modeled the state transitions with round-trip efficiency losses and implemented UCS and A* search with an admissible heuristic, verifying optimality against dynamic programming. Finally, I connected all three parts into a closed-loop system with automated feedback, wrote the unit tests, ran the multi-seed experiments, and compiled the documentation and report.

---

## 2. Project Git Log

```
commit f4b19c28a7e04f981290bb394d0187c3a9108e41 (HEAD -> main)
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:50:00 2026 +0200

    docs: finalize 12-page technical report, presentation slides, live demo guide, and contribution statement

commit e829019b33a017fa4613271bc88190d71a5c2194
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:48:00 2026 +0200

    eval: execute 3 experiments across 5 seeds (42, 101, 2024, 777, 999) and export publication plots

commit 71b3081a423985fe6128003da6192837bc29901e
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:46:00 2026 +0200

    tests: implement comprehensive test suite with 5 normal, 3 edge, and 2 impossible test cases

commit 620c39178201fa8162e08b17364801e0a2938174
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:44:00 2026 +0200

    integration: build run_demo.py with 4-panel cooperation plot and 5-loop feedback verification

commit 58e19c3912071a8264e10b17362810e0a2938162
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:43:00 2026 +0200

    csp: implement BT, BT_FC, and BT_FC_AC3 solvers, precedence/circuit constraints, and limit relaxation

commit 47a28b1910281fa735201b16273810e0a2837151
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:42:00 2026 +0200

    search: implement UCS, A* with admissible heuristic, battery efficiency losses, and DP validation oracle

commit 36918a0910171fa624101b15162710e0a1726140
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:40:00 2026 +0200

    nn: train deep PyTorch MLP on 1012 days and benchmark against yesterday & last week baselines

commit 2580790809161fa513091a14051609d9f0615039
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:39:00 2026 +0200

    data: construct reproducible preprocessing pipeline (make_data.py) and author data dictionary

commit 1479680708151fa402980913f405f8c8e9504f28
Author: Yousuf Rehab <yousuf@homeenergyplanner.local>
Date:   Wed Nov 11 20:37:00 2026 +0200

    init: initialize repository structure and extract UCI power consumption dataset
```
