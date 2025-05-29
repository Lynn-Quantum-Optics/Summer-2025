# 05/01/2025
MP: Stuart Ria

Checked BCHWP and BCQWP angles for HDVA (BCHWP = 67.5, BCQWP = 45). Counts seemed good at around ~1600 for both HD and VA bases. 
For measuring, we set Bob to measure V amd Alice's plates as follows: AHWP = -37.5, AQWP = -120. By hand, it looks like our minimum occurs with the QP around -21 degrees, which is about as expected based on the phase sweep we ran for this state without the phase shift. Before measuring the state we just have to minimize the QP angle and ratio tune. 


# 05/20/2025
MP: Ria

I created a QP sweep file that just runs a QP sweep in the correct basis while measuring in the basis that expects minimum counts (basic_qp_sweep.py). I used this to find the QP minimized counts in the correct basis (using settings from 5/1) at -21.288 degrees for the HD+e^-ipi/eVA state.
I created another ratio tuning file so as not to clog up the phi_plus calibration one that I will use for these two states. It is called "basic_ratio_tuning.py". I ran this file and found the UVHWP should be at -66.02461563913445.

I also took a look at documentation while the files were running to make a note of what is missing and how to better organize it in the future. Additionally, I think the file tree in calibration (especially) needs some cleaning up. I think all important calibration and intro documentation could get moved to also be in the drive so it doesn't get lost when repositories change?
Note: the QP seems to error much more if it hasn't been used recently -- you have to move it a few times using the manager before running a file if it has been a bit, otherwise it errors almost immediately.



# 05/21/2025
MP: Ria

I started by double-checking the value of counts at the given QP angle. It was 44.
I also (finally) synced this computer to github. All conflicts I kept "their" copy.
I checked state purity but accidentally used the wrong PCC angle. Will redo first thing tomorrow.


# 05/22/2025
MP: Ria

I ran the purity check on phi_plus (result in purity_test_05222025.csv) and found the state purity is 0.9391+/-0.0006. This is a little lower than I got when I accidentally ran the sweep with the PCC at zero degrees, so I reran that state and it gave nearly the exact same results (results in purity_test_052222025_2.csv), so I decided to leave the PCC where it is. For some reason, our state purity is a little low today.
It looks like the room is a little warm today (69.1F compared to a max of 66F yesterday, so that is likely it).
Ran into some roadblocks with data processing and which files to run (MORE INFO).

Started process of calibration for the HRVL state - locations for Bob and Alice's plates are below. I also ran the QP sweep and ratio tuning. It looks like counts aren't as minimzed as expected -- investigate tomorrow.
To create the HR+e^(-ipi/6)VL state:
Make phi plus, make meas_basis VV and move as follows:
BCHWP: 0
BCQWP: -45
AHWP: -30
AQWP: -105
QP: -19.137
UVHWP: -112.74443676597194


# 05/23/2025
MP: Ria, Iz, Prof Lynn

I created the HDVA full tomo at different eta values file and ran it. We ran into some errors with data processing and ended up creating a new file for non-mixed states that standardizes file format. Continued standardizing file format (especially file names/editing them) to make them more user friendly as I encountered them.

I also finished up the QP and ratio tuning for the other state. It turns out I sent Alice's HWP to the wrong spot yesterday. Updated values for the QP and UVHWP are listed above. Note the ratio tuning plots had higher uncertainty than for the other state. Counts are still a bit high though. I ended up turning the UVHWP to about -112 rather than -66 and this allowed the qp to reach a minimum within the desired range.