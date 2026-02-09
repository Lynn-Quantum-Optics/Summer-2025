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

# 05/27/2025
MP: Ria

I ran the tomography on the balanced version of the hrvl state, and like with the hdva state, it indicated that the theoretical and experimental rhos did not agree. I verified we were correctly making HRVL and HDVA without the phase shifts and we were. The temperature in the lab is similar to when I ran the calibration on this state, but perhaps that is part of the phase error.

I then ran a purity check to make sure the phi_plus state is still being made as calibrated, and it was a bit low. I also double checked the state calibration for both of our target states, and they no longer seem to be correct. This does make some sense though because the room is significantly warmer today than last week (72F). I started recalibration for phi_plus but then decided to hold off on recalibrating these states as well as phi_plus since F&M said they were coming in to fix the AC today -- though they still hadn't come in by the end of the day.
Looking at the output rho matrices, I deduced what states we appear to be making. They appear to be HD+e^-5ipi/6VA and HL+e^13ipi/12VR rather than HD+e^-ipi/3VA and HR+e^-ipi/6VL. However, when double checking these states by hand, we do still appear to be making some superposition of HR and VL for that state rather than the HL+VR mix implied by the density matrix. I noticed also that some of Stu's files had the definitions of R and L swapped, as compared to the experimental data processing file. It will be important to confirm which convention we are using.
Because the most of this data was gathered last week when the temperature in the lab should have been comparable to the temp during calibration, it appears there is some error in our process of finding the QP angle. However, data gathered today is very likely influenced by this temperature increase.

If the thermostat is not fixed tomorrow, I will need to decide whether recalibrating for the higher temperature and collecting data at that point is worth it or whether I should wait until our AC actually works to collect this data as the high temp impacts laser stability and state purity.


# 05/28/2025
MP: Ria

It looks like the AC issue is affecting many of the nearby labs, so we decided to put a temporary hold on collecting data to see if F&M fixes this issue soon. 
The misdefinition of R and L polarization was actually correct in Stu's file, so I corrected the experimental data processing to have R be (0, i) and L be (0, -i). After doing this correction, I recalculated which state the HR+e^ipi/6VL density matrix corresponded with - it looks like it is actually HR+e^-11ipi/12VL, so the issue from yeterday where the density matrix corresponded to a superposition of HL and VR was corrected by fixing the defintions of R and L.
I also analyzed the csv from Thursday (the AC worked somewhat Thursday morning around the time of data collection) to get a density matrix and compared it with Friday's data -- it looks like Friday's temperature was not the cause of the pi/2 phase shift in the state we are creating. I then spent some time reviewing our process for making these states to determine what could be the cause of this inconsistency. 


# 05/29/2025
MP: Ria

I worked on reviewing the process used for state generation and identified a potential issue. As the AC is still not fully working, I recalibrated phi_plus for the warmer lab temp, and started recalibrating the two states I am working on so that I can run a tomography tomorrow and verify whether or not the change I made to the procedure work.
Warmer Phi_plus calib: QP ANGLE=  -16.8


# 05/30/2025
MP: Ria

The AC is working again, note the lab is about 58F due to the way the building is cooled (as our air vent is currently propped open while waiting for parts on order, there is no way to warm the lab). 

Since the lab temperature has fluctuated so much in the past week, I spent some time recalibrating the phi_plus state as well.
The new QP angle is -14.943 and the UVHWP angle is -65.06993022717926. I then double checked the calibration of Bob's creation waveplates for phi_plus as the density matrix from the tomographies last week showed an over emphasis of H and V as compared to D and A. The current phi_plus state purity is: 0.9514+/-0.0007. I collected some detailed data on counts while making phi_plus over 30 seconds. The results are below.
- HH Counts: 1507+/-7
- VV Counts: 1500+/-6
- HV Counts: 16.3+/-1.0
- VH Counts: 16.0+/-0.5
- AD Counts: 40.3+/-0.7
- DA Counts: 32.1+/-1.0
- DD Counts: 1496+/-4
- AA Counts: 1483+/-7

We determined part of the phase issue is due to the fact that our measurement basis calculations did not take into account negative phase shifts (such as the one we want here), so there were some sign issues.
FOR HD+e^-ipi/3VA, set Bob to measure V and AHWP @ -7.5, AQWP @ -60
FOR HR+e^-ipi/6VL, set Bob to measure _ and AHWP @ -15 and AQWP @ -75


# 06/02/2025
MP: Ria

It looks like something in the set-up got bumped since Friday. We adjusted the mirrors and the BBO as counts were lower than usual. I then recalibrated the UVHWP, ran a purity check (purity is now 0.9408+/-0.0009), and double checked the counts. The counts are as follows: 
- HH Counts: 1486+/-7
- VV Counts: 1450+/-11
- HV Counts: 14.4+/-1.0
- VH Counts: 14.7+/-0.5
- AD Counts: 33.8+/-1.3
- DA Counts: 56.9+/-1.4
- DD Counts: 1444+/-5
- AA Counts: 1431+/-9

I calibrated the HD+e^-ipi/3VA state with the updated measurement basis. The calibration is as follows: UVHWP @ -115.2803939016242, QP @ -27.027

I then collected data on the counts rates. Note HA and VD are not balanced as well as one would hope. Counts are as follows:
- Min counts: 50.2+/-2.6
- HD Counts: 1479+/-5
- VA Counts: 1478+/-5
- HA Counts: 11.9+/-0.4
- VD Counts: 17.9+/-0.6
I then ran a full tomography for the balanced version of this state (chi=90)


# 06/03/2025
MP: Ria

I analyzed the data from yesterday. The phase shift appears to be 70 degrees, rather than the target value of 60 degrees. I decided to recalibrate the QP angle, this time by finding the minimum counts beteween the TWO minimized bases rather than picking one. The measurement settings for the second basis are FOR HD+e^-ipi/3VA, set Bob to measure H and AHWP @ -52.5, AQWP @ 30. I recalibrated the QP and re-ratio tuned to get the QP @ -27.379 and the UVHWP @ -115.56962464985094. Counts are as follows:
- Min counts 1: 52.8+/-1.7
- Min counts 2: 37.8+/-1.2
- HD Counts: 1501+/-4
- VA Counts: 1507+/-6
- HA Counts: 11.7+/-0.6
- VD Counts: 17.1+/-0.5
Note the imbalance between HA and VD still exists. 
I then ran another tomography. However, results are similar to before, so I think fixing the imbalance between HA and VD is an important next step.


# 06/04/2025
MP: Ria

As an alternative method for determining the QP angle that produces the correct phase shift, we decided to do a more complicated process that involves measuring a few different bases at different QP angle and using this to calculating the observed phase shift, gamma, in the density matrix. I wrote the code file, and ran it. However, in middle of this, two of the measurement waveplates lost connection from the computer and I had to side-track to recalibrate them as one of them did not error while at zero. I double check counts for phi_plus and hd_negpi_3_va, and they were not balanced in the same way as previously (and visually BQWP/AQWP looked a bit off), so I decided to recalibrate the measurement waveplates before proceeding. It looks like the calibration didn't change as significantly as I would have expected, so perhaps the measurement waveplates did disconnect at their calibrated zeroes. Tomorrow, I will double check phi_plus calibration and update the calibration for hd_negpi_3_va using the new process.

- AQWP: 0.538 -> -0.710
- AHWP: -.465 -> -0.285
- UVHWP: -25.78 -> -26.431
- BHWP: -6.168 -> -5.914
- BQWP: 99.04 -> 99.724
- BCHWP: 0 -> -0.554
- BCQWP: 128.25 -> 127.484


# 06/05/2025
MP: Ria

I double checked the calibration for phi_plus then tested the new gamma_determination method for determining the qp angle to produce the correct phase shift for the hd_negpi_3_va state. There were a few modifications I had to make to the file from yesterday -- instead of finding theta (the ratio of the HD:VA), if we collect two more basis measurements, we can calculate gamma without needing to know the experimental purity.
Note that BCHWP had the error that usually occurs with the qp. I'm wondering if it has something to do with the way it is zeroed since this was never an issue before recalibration (ie does the issue come from where the zero location is relative to the waveplate's hardware home).

UVHWP Update: -64.93242765727796->-64.14652091578432
QP UpdateL-14.9->-15.6788

Phi_Plus counts after measurement waveplate & qp recalibration:
- HH Counts: 1478+/-9
- VV Counts: 1468+/-6
- VH Counts: 9.9+/-0.7
- HV Counts: 11.5+/-0.7
- AD Counts: 29.2+/-1.0
- DA Counts: 27.6+/-1.2
- DD Counts: 1451+/-6
- AA Counts: 1451+/-10
Phi_plus purity is currently 0.9641+/-0.0007.

It looks like the measurement waveplate recalibration combined with the QP recalibration has fixed the imbalance in vh/hv and da/ad counts. Understanding why this is seems like something we should try to do.

While running the gamma_determination file, I noticed an error in the bases we were using for caluclating gamma. I have corrected this in the file. After retuning, for hd_negpi_3_va the UVHWP angle was -114.90445548609685 and the QP angle was -26.773838565224096.

Counts are as follows: 
- Min counts 1: 46.6+/-1.0
- Min counts 2: 37.9+/-0.5
- HD Counts: 1549+/-9
- VA Counts: 1551+/-7
- HA Counts: 9.9+/-0.4
- VD Counts: 14.4+/-0.7

It looks like the recalibration of the measurment waveplates combined with the new gamma determination method has led to more balanced counts in general. There is still a bit of an imbalance in HA/VD (though they are closer than before)


# 06/09/2025
MP: Ria

The lab is a bit warmer this week (~61F rather than mid-50s). I double checked the counts and decided the temperature change warranted recalibrating the QP for the hd_negpi_3_va state. After, I ran the full tomography over all the eta values and analyzed the data.

The QP angle was: -28.13743446751645 and counts are as follows.
- Min counts 1: 55.1+/-0.7
- Min counts 2: 46.3+/-1.2
- HD Counts: 1572+/-4
- VA Counts: 1642+/-7
- HA Counts: 11.3+/-0.5
- VD Counts: 14.3+/-0.6


# 06/10/2025
MP: Ria

The temperature was back down to 56F today, so I used the same QP calibration as last week after double checking counts.
I then ran a couple more full tomographies with the full eta/chi sweep.
- Min counts 1: 46.6+/-1.2
- Min counts 2: 33.2+/-1.1
- HD Counts: 1566+/-12
- VA Counts: 1557+/-8
- HA Counts: 12.3+/-0.6
- VD Counts: 15.30+/-0.30

I ran a couple of sweeps for the hd_negpi_3_va states, and also processed the data with Iz using the new data processing file. There were some issues we found with the file, so they did some debugging & made other changes throughout this.

I also noticed the data from yesterday had a weird fit for the UVHWP -- it looks like the manager wasn't clearing at the right times, leading to the fits being messed up. I fixed this, but because of that, I would not trust the data in the "TRIAL 1" folder as being for the correct chi values (because the UVHWP fit was so far off). I noticed this because the fit for the chi=90 had the UVHWP around 66 degrees, which is the complete wrong quadrant for this state and lead to some sign errors.


# 06/11/2025
MP: Ria

I ran a few tomographies for the hd_negpi_va state. Just for reference, counts at the start of the day were as follows.
- Min counts 1: 46.9+/-1.2
- Min counts 2: 34.8+/-1.1
- HD Counts: 1557+/-6
- VA Counts: 1584+/-10
- HA Counts: 12.5+/-0.6
- VD Counts: 13.9+/-0.7

I also calibrated the hr_negpi_6_vl state in preparation for running a tomography on it tomorrow.
FOR THE HR+e^-ipi/6VL state: QP Angle: -12.11346435546875 UVHWP Angle: -111.29939852262797

 
# 06/12/2025
MP: Ria

We attended the lab safety training in the morning, so I didn't quite have enough time to run the tomography. 
In the afternoon, I ran a tomography on the hr_negpi_6_vl state so we could process this data. It looks like the calibration for the gamma isn't quite as good as for the other state, so I'll want to take more data for this in the future.
For reference, counts were as follows for the hr_negpi_6_vl state:
- Min counts 1: 35.9+/-1.1
- Min counts 2: 37.7+/-0.5
- HR Counts: 1555+/-6
- VL Counts: 1568+/-7
- HL Counts: 16.1+/-0.6
- VR Counts: 16.2+/-0.


# 06/13/2025
MP: Ria

- Min counts 1: 49.4+/-0.9
- Min counts 2: 38.6+/-1.0
- HD Counts: 1488+/-4
- VA Counts: 1555+/-7
- HA Counts: 10.9+/-0.5
- VD Counts: 12.7+/-0.8

I started running one last tomography before the chillers turn off. It looks like they turned the chillers off an hour early, so the quality of the data near the end (last 3 data chi values) is not as before. This is the trial 6 data. However, the front of the lab (with the thermostat) warmed up much quicker than the back, where the experiment was being run.


# 06/16/2025
MP: Ria

I ran a few tomographies today to collect more data. The counts (measured before the tomographies) are as follows.

hd_negpi_3_va:
- Min counts 1: 50.5+/-1.4
- Min counts 2: 38.6+/-0.5
- HD Counts: 1545+/-5
- VA Counts: 1606+/-8
- HA Counts: 11.0+/-0.4
- VD Counts: 14.5+/-0.4

hr_negpi_6_vl:
- Min counts 1: 37.7+/-1.1
- Min counts 2: 36.8+/-0.7
- HR Counts: 1591+/-6
- VL Counts: 1617+/-8
- HL Counts: 16.13+/-0.30
- VR Counts: 16.6+/-0.8


# 06/17/2025
MP: Ria

I processed the tomography data that I recently collected. The hr_negpi_6_vl state shows great correlcaiton with theory/adjusted theory for the W_5; however, it is consistently lower than the W_3 values. I looked more into the data and noticed the W_3 minimized values for theory are NOT actually the minimized values. Iz will look more into this. This state is consistently witnessed by witness W_5_6.

I also looked a bit more closesly at the hd_negpi_3_va data to try to understand why it looks as strange as it does. It looks like the W_5 triplet 3 shows inconsistency in both theoretical and experimental data as to which witness minimizes it (depending on the chi value, some chi have witness 7, some 8, and some 9). Interestingly as well, the triplet 3 values are essentially indentical to the triplet 2 values, and triplet 2 should NOT witness this state. I will look more into this.

Looking back at the code used to find these states last summer, I reran it to find what the witness values should be according to the theory. 
From this, I determined hd_negpi_6_va is witnessed by W_5_8 AND W_5_9, and that hr_negpi_3_vl is witnessed by W_5_4 AND W_5_6.

Note W_5_8 and W_5_4 (respectively) most commonly had the absolute minimum values

After comparing theoretical witness values with both the old and new witness/minimization schemes, I noticed one major issue: W_5_6 is incorrect -- it gives different values in the new data set than in the old one (off by about 1/2). It looks like there was a typo in the new witness file that led to this issue.


# 06/18/2025
MP: Ria

We worked at solving the few issues from earlier this week by checking the witnesses themselves and going through the code step by step. Ultimately, we found the print statement discrepencies (which made it look like the minimization wasn't happening) were due to small issues in the code that affected ONLY the print statements, not the actual plots. Most notably, the adjust_rhos function actually modified the theoretical density matrix into the adjusted theory one, so after adjust_rhos was called in the witness calculation, the theoretical density matrices were no longer correct. Iz fixed this such that the function no longer modifies its arguments. However, we are suspicious this function doesn't do what it is supposed to do, and want to cross check this with Prof Lynn.

Additionally, I think I have identified the reason for the discrepancies between W5 theory and experimental witness values: it looks like something is going wrong in the UVHWP sweep/calibration in the tomography file that leads to the chi values not matching up with what they claim to be. I will investigate this further.


# 06/20/2025
MP: Ria

In investigating the UVHWP sweep in the tomography file, I think I have identified what may be leading to the calibration issue. It looks like the function used to find the optimal UVHWP angle for a given chi value doesn't work very well when the angle it "wants" is outside of the range of data collection. Because of this, I will modify the hd_negpi_3_va file to have a slightly different range the the hr_negpi_6_vl file (as the latter has UVHWP values all within the range of data collection -90 to -135 and the former does not).


Counts are looking a little low, but seeing as my primary objective today is to find a soluation for the UVHWP sweep issues, I will first investigate that (with updated code) and then attempt to improve the counts.

hd_negpi_3_va counts:
- Min counts 1: 50.1+/-1.6
- Min counts 2: 41.1+/-0.8
- HD Counts: 1424.2+/-3.4
- VA Counts: 1498+/-8
- HA Counts: 11.4+/-0.6
- VD Counts: 13.4+/-0.4

phi_plus counts:
- HH Counts: 1412+/-6
- VV Counts: 1427+/-6
- VH Counts: 11.0+/-1.0
- HV Counts: 13.03+/-0.32
- AD Counts: 28.5+/-1.1
- DA Counts: 25.4+/-0.8
- DD Counts: 1419+/-5
- AA Counts: 1402+/-7

- HH Counts: 1436+/-10
- VV Counts: 1459+/-9
- VH Counts: 11.2+/-0.6
- HV Counts: 12.2+/-0.5
- AD Counts: 24.47+/-0.35
- DA Counts: 25.7+/-1.1
- DD Counts: 1442+/-9
- AA Counts: 1411.6+/-2.8

I then recalibrated the quartz plate. The update counts are as follows.

phi_plus:
- HH Counts: 1446+/-8
- VV Counts: 1447+/-7
- VH Counts: 10.8+/-0.8
- HV Counts: 11.33+/-0.30
- AD Counts: 27.1+/-1.1
- DA Counts: 25.9+/-0.5
- DD Counts: 1443+/-8
- AA Counts: 1419+/-5

hd_negpi_3_va:
- Min counts 1: 48.5+/-0.7
- Min counts 2: 37.3+/-1.1
- HD Counts: 1480+/-6
- VA Counts: 1498+/-5
- HA Counts: 11.4+/-0.6
- VD Counts: 13.77+/-0.34

Gamma for hd_negpi_3_va: -1.058+/-0.006.
After checking these values and adjusting the initial UVHWP sweep angle to better encompass the range our chi values have been falling in, I ran another full tomography of the hd_negpi_3_va state.


# 06/23-24/2025
MP: Ria

Note: this entry was compiled from a set of notes left on the lab computer a week after. It looks like I ran out of time these days to fully update the lab notebook, so I am doing my best to fill in the blanks about my thought process from memory.

06/23/2025
Looking at the plotted data for the hr_negpi_6_vl state, there is strong agreement between theory and experiment for W_5 but not for the W_3s. Prof Lynn suggested this may be because the W_5s have more parameters, and are therefore less sensitive to the overall density matrix in that these parameters are "absorbing" some of the imbalances in the density matrices.

Because of this, I decided to look at the impact of a few different aspects of the density matrices on the W_3 and W_5 theory curves for the hr_negpi_6_vl state. I first looked at the impact of changing the phase in the theoretical state from pi/6 to values within the range pi/5.4 and pi/6.6. These graphs are stored in the hr_negpi_6_vl_trial3 data folder in the 2025 repository. As a quick note, all raw data and calibration up to this point has been conducted in the summer 2024 repository, but all processed data and analysis is stored in the 2025 repository as the lab computer hasn't had the new repository for the full summer. This organization structure is something I hope to improve when I have a chance. Interestingly, the phase changes seemed to affect the W_5 curves much more than the W_3s. In order to produce a graph with the W_3 theory and experiment having more agreement, I would have had to introduce a phase shift outside of my tested range -- which was pulled using the most extreme entries in the experimental density matrices. Because of this, I concluded that the phase was not causing the observed differences between theory and experiment.

Additionally, I noticed (by plotting the UVHWP sweep data used in the tomography) that the hr_negpi_6_vl state has the minimum chi get down to ~0.03. However, the hd_negpi_3_va state's minimum chi only gets down to ~0.1. I wonder if this may be the cause of some of the issues with the hd_negpi_3_va state. This data is stored as UVHWP_plot in the hr_negpi_6_vl_trial3 and in the ria_hd_negpi_3_va_UVHWP_test folders located within the ria_hd_negpi_3_va_trials_with_UVHWP_angle_issues folder.

I also realized the imbalance between the diagonal entries of the density matrix for hr_negpi_6_vl was greater than for hd_negpi_3_va. I ran a quick test before I left which changed the density matrix to simulate this imbalance for larger chi values by adding/subtracting. This is the "simulated imbalance" file. It looked like it did lead to greater agreement between theory/experiment for the W_3s and didn't really affect the W_5s, so I determined this would be worth exploring further.


06/24/2025
I decided to double check the ratios between the va/hd and vl/hr counts for their respective states.
For a state of the form cos(chi/2)*|H>|alpha>+e^i*gamma*sin(chi/2)|V>|alpha_perp> I used the formula arctan(sqrt(alpha_perp_counts/alpha_counts)) to determine this value. Results are below.

hd_negpi_3_va:
 UVHWP @ -115.10003140098172
 - HD: 1396.5333333333333+/-13.33274998723902
 - VA: 1378.8666666666666+/-6.199103877891476
 - Value: 44.8 degrees (should be 45 based on the UVHWP sweep data)

 UVHWP @ -128.4243649
 - HD: 2325.0666666666666+/-13.07049263714943
 - VA: 238.06666666666666+/-3.2513245164257474
 - Value: 17.7 degrees (should be 18 based on the UVHWP sweep data)

 UVHWP @ -133.44428684
 - HD: 2458.866666666667+/-13.113436705235628
 - VA: 60.93333333333334+/-0.9510228411791398
 - 8.9 degrees (should be 9 based on the UVHWP sweep data)

 UVHWP @ -135.91511309
 - HD: 2496.6+/-8.787238221167973
 - VA: 48.4+/-2.13020604115606
 - Value: 7.8 degrees or ~.13 radians: this should be zero but is very similar to the minimum value observed in   the UVHWP sweep plots I made
 - HA COUNTS: 12.4+/-0.7774602526460401
 - VD COUNTS: 7.866666666666665+/-0.5734883511361751

 UVHWP @ -137
 - HD: 2501.4+/-8.059087348384363
 - VA: 50.93333333333333+/-0.3858612300930081
 - Value: 8.11 degrees (so we have, as the plots showed, passed the minimum)

hr_negpi_6_vl:
 UVHWP @ -111.29939852262797
 - HR: 1396.0666666666666+/-5.125535202406785
 - VL: 1394.0000000000002+/-6.403124237432837
 - Value: 45.0 degrees (exactly what the sweep file indicates)

 UVHWP @ -133.045283482142
 - HR: 2585.6+/-16.14565645064405
 - VL: 3.0000000000000004+/-0.36514837167011066
 - Value: 2 degrees or ~0.033 radians: this should be zero and is very similar to what we see with this state in the UVHWP plots
 - HL Counts: 16.133333333333333+/-1.485485330343813
 - VR Counts: 8.666666666666668+/-0.5962847939999439

I then decided to try moving the hd_negpi_3_va state the a different quadrant of the UVHWP and QP angle to see if this improved the data. My thought was that we were at a fairly high QP angle, so maybe this was affecting the data somehow. The "new" calibration below was just done by hand as a manner of checking this idea quickly before the end of the day and is not the actual calibration I used in any data calibration.

Original hd_negpi_6_va calibration: hd_negpi_6_va state with QP @ -26.77 & UVHWP @ -115.1000
- minimized counts: 41.46666666666666+/-1.0413666234542205 in basis 1 (bob v, ahwp -7.5 aqwp -60)
- minimized counts: 35.0+/-1.2337837015547826 in basis 2 (bob h, ahwp -52.5 aqwp 30)

New hd_negpi_6_va rough calibration: hd_negpi_6_va state with QP @ -9 & UVHWP @ -65
 - counts in basis 1: 41.733333333333334+/-2.4435857077481673
 - counts in basis 2: 39.733333333333334+/-1.539841261660146
 - HD: 1369.0+/-4.808557187163574
 - VA: 1364.0666666666666+/-5.896703410475314
Note the HD/VA counts are even lower than before. This is a point of interest and something I will investigate later. For now, I just want to quickly test my idea before I levae for the day/

 uvhwp @ -87
 - HD: 2930.0+/-14.690510920696772
 - VA: 37.00000000000001+/-1.8196458751941578
 - Value: 6.4 degrees -> note: based on future data I collected, I believe I misrecorded the HD values as VA values and vice versa (this point should have the VA counts high and HD counts low). IN this case, the value is 83.6 degrees.

 uvhwp @ -43
 - HD:  2558.2+/-14.233606554754642
 - VA: 14.266666666666666+/-0.9683892697555967
 - Value: 4.23 degrees or 0.07 radians which is already smaller than the value with the other calibration.

Because the rough calibration with new QP and UVHWP angles led to a smaller minimum value, I decided recalibrating hd_negpi_3_va in this new quadrant would be a good course of action to see if it led to the data to have less issues (the very non-zero entries at chi~0 and such).
 
 
# 06/25/2025
MP: Ria

Using the information from yesterday, I ran a ratio tuning in the new UVHWP quadrant and found the angle should be -65.18826575028268. I then recalibrated the QP using the gamma determination file to get a QP angle of -7.78401521381579. Note the measurement waveplates DC'd again. I then spent some time double checking the counts (listed below) and then ran a UVHWP sweep to see if the issue of the parameter never reaching zero has been fixed.

hd_negpi_3_va counts:
- Min counts 1: 40.5+/-1.2
- Min counts 2: 32.5+/-1.6
- HD Counts: 1374+/-6
- VA Counts: 1346+/-6
- HA Counts: 10.1+/-0.6
- VD Counts: 12.2+/-0.9

I'm not really certain why the HD and VA counts are so low right now. I will double check phi_plus counts (below):
- HH Counts: 1350+/-7
- VV Counts: 1337+/-11
- VH Counts: 10.2+/-0.7
- HV Counts: 11.1+/-0.7
- AD Counts: 25.1+/-1.5
- DA Counts: 26.0+/-0.6
- DD Counts: 1333+/-5
- AA Counts: 1322.2+/-1.4

Note: the UVHWP is having movement errors similar to the qp today: 
 - The error: "RuntimeError: Sent instruction "b'0maFFFF50D6'" to ElliptecMotor-C_UV_HWP expecting response length 11 but got response b'' (length=0)"
 - Warning when shutting down/moving immediately following the error: "Warning: ElliptecMotor-C_UV_HWP found non-empty com queue. Flushing -> b'0POFFFF50CF\r\n'."

I also ran checkb to double check the measurement waveplate calibration as they dicsonnected multiple times today and I had to power cycle them upon each disconnect. It looks like the calibration is still fine.

I then ran the UVHWP sweep (data stored in the ria_hd_negpi_3_va_test2 folder). I had many issues running the file due to the UVHWP erroring out sometimes, and I'm not really sure why as I'd never had those errors before. Below are the proposed UVHWP angles based on the sweep data, listed from chi~0.05 to chi=90.

[-44.00413891]
[-47.73038604]
[-52.19315399]
[-56.53635221]
[-60.9204664]
[-65.39341473]


# 06/26/2025
MP: Ria
test3
I checked the counts quickly in phi_plus, and they look low (as they were yesterday).
I attempted to recalibrated the mirrors to fix the decreased counts. It was a bit difficult without a second set of hands, but I was able to get it back to where they were before the decrease (so ~1450 HH & VV for phi_plus).
After adjusting mirrors and before ratio tuning, counts were as listed below -- so it looks like I managed to fix the mirror misalignment enough to get us back to where we were before.
- Min counts 1: 41.6+/-0.8
- Min counts 2: 32.0+/-1.1
- HD Counts: 1489+/-7
- VA Counts: 1376+/-7
- HA Counts: 11.5+/-0.8
- VD Counts: 15.1+/-0.8

I then recalibrated the UVHWP and QP for the hd_negpi_3_va state since I moved the mirrors around. I did not spend the time to recalibrate the QP for phi_plus as that file takes a good amount of time to run and I wanted to investigate the hd_negpi_3_va state first.
- UVHWP @ -65.72638622083161 -> -65.42683049252159
- QP @ -9.71449207491852

Note: the measurement waveplates disconnected again, but the UVHWP is no longer giving those errors.

AFter recalibration, the counts were as follows for hd_negpi_3_va.
- Min counts 1: 49.1+/-1.8
- Min counts 2: 32.4+/-0.8
- HD Counts: 1409+/-4
- VA Counts: 1497+/-10
- HA Counts: 11.0+/-0.6
- VD Counts: 13.9+/-0.4
The ratio tuning says this is where hd and va should be balanced, but they clearly weren't. I figured it was a ratio tuning issue, and decided to go ahead and run the tomography since I was running low on time and the file would recalcaluate a value anyways.

I ran the tomography file to collect UVHWP sweep data. These results are ria_hd_negpi_3_va_test3.
- [-43.76538381] chi~0
- [-39.85422415] chi~18
- [-52.11162297] chi~36
- [-56.46572728] chi~54
- [-60.84917988] chi~72
- [-65.30470434] chi=90 (different from the hd/va balanced calibration I had found, I later confirmed this is the value that produced balanced counts)

After this, I ran the tomography with a slightly adjusted range to hopefully avoid the second chi value being in the wrong quadrant -- which it was in the test3 data listed above. This is hd_negpi_3_va_trial9. Unfortunately, the UVHWP sweep still led to the second chi value being in the wrong quadrant (because it swept past the minimum), so I used the UVHWP sweep data (minus the angles that are in the incorrect quadrant) to recalculate the UVHWP angle for that chi value. I then collected data on that point immediately after the full tomography, and stored it in the same folder. This is noted in the folder as well (in file names).

Using the idea I had Monday for the hr_negpi_6_vl state, I calculated what the actual state we are making for hr_negpi_6_vl is -- it appears to have an ~42 degree difference between the H/V in R and L, rather than 45. I found this by assuming the state we were making was cos(chi/2)*|H>|R'>+(e^i*gamma)*sin(chi/2)|V>|L'> where R'=cos(theta)|H>+i*sin(theta)|V> and L'=sin(theta)|V>-i*cos(theta)|V>. I found that cos^2(theta) was the first (and last) entry in the density matrix, both of which were ~.275 and that sin^2(theta) were the middle entries on the main diagonal, both ~.225 (these values rounded so the diagonal sums to 1). Thus, theta=arctan(sqrt(.225/.275))


# 06/27/2026
MP: Ria

Today, I looked a bit more into the phase drift with the hd_negpi_3_va state by running the gamma_determination file with each of the UVHWP angles calibrated yesterday. This is all_gamma_chi_06272025_trial1 and shows ~1.5 radian drift over the full range of chi values. With Prof Lynn, we decided that since previous analyses in the lab determined we are better at making primarily vertically polarized states than horizontally polarized, we should swap the sine and cosine to change the state from cos(chi/2)*|HD> + sin(chi/2)*e^(-ipi/3)*|VA> to sin(chi/2)*|HD> + cos(chi/2)*e^(-ipi/3)*|VA> (so we can get closer to making the target chi=0 value). Additionally, she determined that this phase drift may be a product of the UVHWP not acting as well as it should, and that shifting the quadrant we are using should fix this issue.

I ran a UVHWP calibration for this (angles are below, sweep is ria_hd_negpi_3_va_test4) and used these angles in the gamma_determination file to characterize the phase drift in this section. This data is all_gamma_chi_06272025_trial2. Note that the drift does not seem to improve.
- [-88.40268658] ~chi0
- [-84.76152667] ~chi18
- [-79.38040221] ~chi36
- [-74.5211473] ~chi54
- [-69.83436021] ~chi72
- [-65.26673128] ~chi09

I then ran another trial in a different quadrant of the UVHWP (angles used are below, sweep is ria_hd_negpi_3_va_test5), and found much the same thing -- no improvement in the drift. This indicates the explanation we had settled upon for the drift is likely not what is causing it. (all_gamma_chi_06272025_trial3 stores this data)
- [-178.43334997] ~chi0
- [-174.4035695] ~chi18
- [-169.28190979] ~chi36
- [-164.47610195] ~chi54
- [-159.79179246] ~chi72
- [-155.20270599] ~chi90

Using the math I did yesterday for the hr_negpi_6_vl state, I remade all of the hr_negpi_6_vl plots with an adjusted theoretical state that more closely matched what we made, and the theory/experiment discrepancy from before no longer present in the W_3s with minimal effect on the W_5s.


# 06/30/2025
MP: Ria

I generated plots of phase diff in the hr_negpi_3_vl state using the same method as yeserday, and confirmed it is much smaller than in the hd_negpi_3_vd state.

I also looked at last year's data. Using the tomography datas and analyze_phase, I determined there was not as large of a phase drift in the mixed data as we are seeing now. Unfortunately, it was unclear what unmixed data was used to generate these mixed data sets. I contacted Lev to ask where the unmixed data used in generating the paper state mixed sets was, but he didn't respond yet.

I ended up analyzing one unmixed hdva data set, and did find it had drift similar to what we are seeing with hd_negpi_3_va.


# 7/1/2025
MP: Ria

Since Lev still hasn't gotten back to me about the unmixed data corresponding to the mixed data used in the paper states, I decided to just analyze all data I found last year to find the phase drift. All of this drift data is located within the Summer2024 repository, directly in the raw data folders in framework and Lev's folder as phase_drift_data.

It appears that all states of the form hdva/havd have more phase drift than hrvl/hrivl states do -- something that aligns with what we ahve observed. However, states of the form havd seem to have slightly less phase drift than states of the form hdva.

They are fixing ac tomorrow morning so will likely need to recalbrate the set up.


# 07/02/2025
MP: Ria

They apparently fixed the AC in the lab this morning by removing the zip ties and replacing the mechanism that allows the air vent to open and close. However, it is not any warmer in the lab.

phi_plus counts:
- HH Counts: 1472+/-7
- VV Counts: 1451+/-6
- VH Counts: 11.5+/-0.5
- HV Counts: 12.5+/-0.6
- AD Counts: 30.7+/-0.9
- DA Counts: 38.9+/-1.5
- DD Counts: 1444+/-7
- AA Counts: 1430+/-5

I calibrated the state ha_vd with a zero phase (QP @ -16.1584427682977) so that I could test the phase drift in the state.
ha_vd counts:
- AH counts: 28.8+/-1.1
- DV counts: 29.1+/-0.9
- HA Counts: 1430+/-6
- VD Counts: 1423+/-5
- HD Counts: 10.90+/-0.32
- VA Counts: 11.2+/-0.7

I then ran a UVHWP sweep (ria_ha_vd_test1) and got the below angles for the UVHWP. I then ran the gamma determination file on each of these angles, and found the drift data stored in ha_vd_phase_drift_test. Drift seems smaller with the havd state, so I will proceed with finding if a variation of this state is witnessed by W5_t3. Drift is about 0.2 vs 0.5 with the the other state.
- [-87.73224977] ~chi0
- [-92.38054896] ~chi18
- [-97.20772917] ~chi36
- [-101.92845518] ~chi54
- [-106.56321264] ~chi72
- [-111.11112706] ~chi90


# 07/03/2025
MP: Ria

I reinstalled the venv on my laptop bc it broke and scipy.optimize no longer works. I then created a version of Lev's state finding file called gen_pure_state_gamma that finds the witness values for a user defined state and tested it on knownt states (hd_negpi_3_va, hr_negpi_6_vl). I then found that the state ha_negpi3_vd is witnessed by t3 and decided to move on with this state. Note: all this data is in the summer 2025 repository under ria/w_5_triplet_3_state_finding.

phi_plus counts at the start of the day:
- HH Counts: 1411+/-6
- VV Counts: 1377+/-6
- VH Counts: 10.1+/-0.4
- HV Counts: 11.1+/-0.7
- AD Counts: 28.9+/-0.9
- DA Counts: 34.5+/-0.8
- DD Counts: 1383+/-9
- AA Counts: 1368+/-9

Note: wavplates dc'd again 3 times today
Note: measuring min counts for ha_negpi_3_vd is same waveplate settings as hd_negpi_3_va

I sweep around -9 qp angle for gamma and found QP: -7.165756064967105 for the desired phase.
I found the UVHWP angle was -65.86116630152654 for the balanced state.
 
ha_negpi_3_vd counts:
- Min counts 1: 29.8+/-0.6
- Min counts 2: 27.0+/-1.4
- HD Counts: 10.8+/-0.5
- VA Counts: 11.8+/-0.6
- HA Counts: 1417+/-12
- VD Counts: 1421+/-5

I then ran a tomogrpahy. Note that I modified the ria_hdva file to create ria_havd and unintentionally left the flipped sine and cosine in the data, so ria_ha_negpi_3_vd_trial1 was found on the state sin(chi/2)*|HA> + cos(chi/2)*e^(-ipi/3)*|VD> rather than the intended cos(chi/2)*|HA> + sin(chi/2)*e^(-ipi/3)*|VD>. I processed this data and realized there is now a phase drift. Will look more into that after the holiday. I also expanded use of gen_pure_state_gamma for multiple gamma/state tests.

# 07/08/2025
MP: Ria

I rewrote a file to read the tomography data and calculate gamma the same way as the gamma determination file (this is called gamma_calculation.py). We discussed how purely we can make something like |V>|alpha> and how we should check this with and without the UVHWP.
-> we should be better at |V>|alpha> than at |H>|alpha> if we are correct in what we know about the set up
-> no UVHWP should be ONLY |V>|alpha>

We also discussed the issues Isabel noticed with the W_5 code and how to check that the new code written over the summer matched previous code.
-> pull the alpha value and the theta value it is minimizing for W_5_8 on the basically no entangled state and calculate the new pure state that is rotating the first particle by -alpha about the y axis. calculate W_3_6 should have basically same values. If they are not the same, WE HAVE ISSUES WITH THE W_5 logically.


# 7/9/2025
MP: Ria

Since I flipped the ratios from when I had found ha_vd as having less phase drift than hd_va, I checked
|H>|alpha> versus |V>|alpha_perp> counts for a state of the form cos(chi/2)|H>|alpha>+sin(chi/2)|V>|alpha_perp> to see which one minimizes more as chi changes.
I just used the current phi_plus calibration for simplicity -> note the qp angle produces some phase shift but it may not be exaclty zero as I have not recalibrated phi_plus recently (but it should be ~0 as that's where it was calibrated to)

phi_plus counts:
- HH Counts: 1421+/-7
- VV Counts: 1330+/-4
- VH Counts: 11.2+/-1.1
- HV Counts: 12.8+/-0.6
- AD Counts: 30.8+/-0.7
- DA Counts: 38.4+/-1.2
- DD Counts: 1368+/-7
- AA Counts: 1339.5+/-2.9

phi_plus: 
- 2.500000416666725e-07: [-42.90096457] -> [0.03210369387100021+/-0.0019799547380766115]
- 3999999.333333252: [-87.72854883] -> [1.5185091827060833+/-0.0017540315726145511]

Waveplates keep disconnecting

we are better at making HH than at making VV
-> note this is not pure phi_plus, I think there is some phase shift there I just didn't calculate it

As a follow up from this discussion yesterday, I did some data collection to check how well we were making |H>|alpha> as compared to |V>|alpha_perp>.
For simplicity, I used the phi_plus calibration to do so, but without verifying the QP angle still produces a zero phase as the phase shift is shouldn't affect this. So, the state I ended up creating was cos(chi/2)|HH>+(e^i*gamma)*sin(chi/2)*|VV> where gamma is presumably near zero, but not necessarily exactly zero.

I then swept the UVHWP over a large range of data point (>45 degrees in total to ensure I enclosed both the maximum and minimum chi), and found where this data set placed chi=0 and chi=pi. I then used a different file to take data at both of those UVHWP angles (for a bit longer than the previous file) and to calculate chi/2 for both points.

Ultimately, I found that where chi/2 should have been zero, it was 0.0321 and where chi/2 should have been pi/2, it was 1.518. To me, this makes it appear as if we are currently better at making |H>|alpha> than we are at |V>|alpha_perp>, which is not what we were expecting. As a quick note, cos(0.0321)~0.9995 whereas sin(1.518)~0.9986.

I wasn't really sure why this is -- I know that we said the only two components that should affect this balance are the UVHWP and the BBO, but as the UVHWP is still in, it should offset any effect from the BBO. 

I do wonder if this could partially explain why the ha_negpi_3_vd tomography data looks so much different from what we would expect -- since there appears to be more extraneous |H>|alpha> counts when making |V>|alpha_perp> than vice versa and I unintentionally ran the state with majority |V>|alpha_perp> rather than the |H>|alpha> I had intended (and also verified phase drift for). We had previously (with hd_negpi_3_va) discussed switching to making primarily VA rather than HD as we are supposed to be "better" at that, but it didn't actually improve anything. Perhaps now with this state, if we switch to making primarily HA rather than VD (which it currently appears we might be better at), maybe this would improve the quality of the experimental density matrices? I am just wondering if somehow the slight difference in the produced extremal chi values is playing a role in the phase drift we are observing. At the least, it would hopefully allow us to improve the chi=0 point? 

Now that I have the process for generating this data down pretty well (it took a quite a bit longer than I expected as I ended up having to write some new files), I am trying to decide whether to repeating this data collection with alpha = D and alpha = R to see if the results are consistent with the phi_plus results as well as the chi values I observed when generating tomography data the past few weeks is worth my time. The data I have collected so far does make me curious to check whether going back and taking data for the ha_negpi_3_vd state with the ratios as we had initially intended (cos(chi/2)*|HA> rather than sin(chi/2)*|HA>) leads to different results as compared with Friday's data.

Finally, I wanted to make a note that the connection error with the measurement waveplates has been occurring much more frequently in the past two weeks. Because of the way the manager throws these errors, it is unclear if one waveplate is repeatedly disconnecting, or if it is an error with multiple of them. Today, I ended up having to power cycle the measurement waveplates between every file or two because of this -- something that I think is worth noting.

We ultimately decided the next step is to remove the UVHWP to see how well we can make VV without it.



# 7/10/2025
MP: Ria

In line with what Prof Lynn recommended, I removed the UVHWP to check how well we are able to make VV without it in. I also reanalyzed the old data to find the produced chi values.

no UVHWP: [1.5289762635044757+/-0.0014746582518083504] & [1.5243398331404863+/-0.0013936876474972514]: they are different BUT they both given (sin(chi))~0.9990 vs 0.9991 other: 0.0418, 0.0468
- for cosine 0.0321-> 0.9995 other: 0.0321
- for sine 1.518->0.9986 other: 0.0528

I noticed that we have more excess HH counts "leaking" in than VV counts, but I'm not sure why that is.
-> measurement bases not well lined up: nope would impact both
-> pump beam is horizontal: we are picking up some pump beam light?
-> we have filters and it is much different wavelength BUT it is also much brighter

To realgin uvhwp when putting it back inL: swap razor for paper folded once bc razer was poor aligned


phi plus count rates:
- HH Counts: 1480+/-9
- VV Counts: 1228+/-6
- VH Counts: 15.6+/-0.8
- HV Counts: 13.4+/-0.7
- AD Counts: 128.5+/-2.2
- DA Counts: 113.5+/-1.7
- DD Counts: 1254+/-5
- AA Counts: 1227+/-4

 recalib phi_plus:
- new UVHWP -64.14652091578432->-65.39653496993216
- new QP ->
- had to readjust mirrors/bbo due to low counts-> couldnt myself so asked for help in moring

We decided that because we had more HH counts leaking than VV there was some source of noise that appeared to be muddling our low chi values. Thus, we decided to (at least for this state if not others) take only 5 data points (just removing the chi~0 points as it seemed to be dominated by noise).



# 07/11/2025
MP: Ria

Unfortunately, we didnt' get mirrors much better with two people so we just called it good despite the lowered counts :((. I then recalibrated phi plus and ha_negpi_3_va and ran a tomography.

phi_plust recalib
- readjust UVHWP -65.39653496993216->-64.5164626272101
- QP -15.6788->-14.993

phi_plus counts:
- HH Counts: 1325+/-5
- VV Counts: 1320+/-8
- VH Counts: 12.2+/-0.5
- HV Counts: 12.3+/-0.6
- AD Counts: 23.8+/-0.8
- DA Counts: 29.8+/-1.5
- DD Counts: 1300+/-10
- AA Counts: 1308+/-5

ha_negpi_3_va recalib:
- UVHWP -65.86116630152654->-65.937744140625->-66.30258419639185
- QP: -3.8691303453947365
- gamma: -1.060+/-0.005

counts:
- Min counts 1: 30.3+/-1.0
- Min counts 2: 29.4+/-0.7
- HD Counts: 13.8+/-0.4
- VA Counts: 15.5+/-0.6
- HA Counts: 1342+/-6
- VD Counts: 1335+/-8

While the tomography was running, I made the following notes:
- hr_negpi_6_vl: biggest drift using the same gamma calculation method as the file I use to calibrate it was 0.04 (from chi~18 to chi=90)
- ha_negpi_3_vd & hd_negpi_3_va data (comparing today's data with the last collection of hd_negpi_3_va) has a drift of ~0.4-> 10x larger... the plots look very similar as well... with this in mind, I am inclined to believe there is something going on with the states in which bob is diagonal/antidiagonal that is SOMEHOW not occurring when bob is circularly polarized.... could it be some "contaminating" light from somewhere that is just not circularily polarized????????
- note before I fixed the ratios for the ha_negpi_3_vd data this drift was about 0.7
- these drifts are smaller than the ones I had plotted before, likely because this is essentially using 2 density matrix entries and the plots I had made earlier only used one... perhaps they "cancel" each other out a little bit... anyways, I would believe these drift calculations more directly correspond to the way phase impacts witness values as hr_negpi_6_vl has essentially no drift with this (0.04) and the witness values show phase agrees whereas the other method I used (using 1 density matrix entry) showed drift on the order of 0.2 for this (which we would maybe see on the graph)
- in contrast, the ha_negpi_3_vd and hd_negpi_3_va show phase drift on the order of 0.4 with both methods and their graphs reflect this trend
- I was hoping to look at old data to confirm that the mixed data showed essentially no phase drift as compared to the unmixed data using this method, but as it relies on the counts data, that is a bit more difficult to do as the csv i was using for generating this data are not created for the mixed data sets
- additionally, though we decided not to go down to chi=0.001 radians due to the noise, the UVHWP sweep I conducted on the ha_negpi_3_vd state would suggest that the minimum chi is ~0.05, which would be on par with the hr_negpi_6_vl state AND what I would hope we are seeing :)... I did take this point, but I think the UVHWP minimized into the wrong quadrant as the phase data is competely off AND the "actual" chi I calculated is larger than this value the sweep would suggest (looking at the UVHWP sweep, it does look like this minimized past the minimum)


# 07/14/2025
MP: Ria

I ran a UVHWP sweep using ria_havd to find each UVHWP. I then went through and for each angle found the proper gamma (using the gamma calibration file in state_calibration_code), and then input the gamma and UVHWP angle into the tomogrpahy file and ran a tomography using that data. Note that for some data points, it is necessary to do an iterative process if the gamma is far enough from the QP angle used to find UVHWP angles.

COUNTS:
- Min counts 1: 39.9+/-1.2
- Min counts 2: 26.1+/-0.6
- HD Counts: 13.4+/-0.6
- VA Counts: 14.7+/-1.1
- HA Counts: 1364+/-4
- VD Counts: 1333.4+/-3.2

- Min counts 1: 28.3+/-1.0
- Min counts 2: 29.8+/-1.0
- HD Counts: 11.2+/-0.9
- VA Counts: 13.5+/-0.4
- HA Counts: 1331+/-4
- VD Counts: 1339+/-4

UVHWP prelim sweep in folder
qp and uvhwp data stored in folder
- [-44.55908126]
- [-48.60257788]
- [-53.14131287]
- [-57.58839513] -3.114393022938778
- [-62.04858616] -3.9992205521934907
- [-66.55361065] -4.827486214637755

-> other chis not able to get needed gamma in this quadrant, moved to alternate quadrant w UVHWP at -112.5 not -67.5

- [-132.55902792] UNUSED DATA POINT
- [-128.59918009] -24.251088513826073 -> [-132.50543769]
- [-124.24155098] -25.20815734863281 -> [-127.38849844]
- [-119.93580218] -25.603982222707646 -> [-122.98117298]
- [-115.58783425] -25.831407406455597 -> [-118.51952741]
- [-111.16360422] -26.00327309056332 -> [-114.11387723]

 [-132.50543769, -127.38849844, -122.98117298, -118.51952741, -114.11387723]: set of UVHWP angles used


# 07/15/2025
MP: Ria

I continued to go point by point... first check chi=90, then 72, then 54, then 36, then 18 (doing the tomography immediately after calibrating).

- Min counts 1: 28.6+/-0.5
- Min counts 2: 36.1+/-1.3
- HD Counts: 10.8+/-0.7
- VA Counts: 12.0+/-1.0
- HA Counts: 1293+/-5
- VD Counts: 1289+/-8


90: -25.69667326274671 & [-113.97057031] -> -25.713055098684208 & [-113.85453091] -> -25.593876246402132 & [-113.88752972]
not really muhc change in the chi checks here so probably don't need to calibrate quite that much (didn't seem to be changing calibration that much and i had to go)

Note there was a lot of error in phase at smaller chis (maybe I should have double check the UVHWP & gamma both once more)/


# 07/16/2025
MP: Ria

- HH Counts: 1308+/-8
- VV Counts: 1308+/-6
- VH Counts: 13.1+/-0.8
- HV Counts: 14.3+/-0.7
- AD Counts: 27.2+/-1.1
- DA Counts: 28.9+/-0.5
- DD Counts: 1296+/-4
- AA Counts: 1291+/-4

For today, i am calling the 0.99X phase "close enough" to the target phase of 1.04 as there is some error & calibration isn't improving it *that* much. Besides, the actual phase in the density matrices is a bit different from the value shown when calibrating. I will calibrate to about this phase and if it is still bad in the plots, will rerun the calibration *another* time for even a tighter phase. I probably will need to rethink the gamma fitting tho bc a value spit out by the fitting doesn't always have the exact same phase when you run the gamma check which implies maybe the fitting could use some work?????

- 72: -25.482613814504525 & [-118.23365587]
- 54: -25.05831660220498 & [-122.62702233] -> -25.10857029965049 & [-122.54623276]
- 36: -24.533135665090455 & [-126.90093284] -> -24.669723510742188 & [-126.9772774] -> -24.638519287109375 & [-126.96943523] -> -24.312966469212583 & -126.82651768
- 18: -18.96165527343748 & -129.26352958 -> -23.4084573203639 & [-131.69597203] -> -21.374077405427627 & [-130.44066667] -> -22.757037032277957 & [-131.29599484]... ended up just going with [-130.44066667] & -22.757037032277957 bc it gave chi=0.3542291860031217~20 degreees and gamma (at this ratio, the gamma is very sensitive to the chi, so ratio tuning AFTER gamma calculation significantly alters the produced chi.... we will place this poijnt correctly in the plots to correspond to the fact that it isn't exactly chi=0.314...~18 degrees)

 to be fair, our previous data from this summer has the chis within 0.01 radians of the target for hr_negpi_6_vl, but the data from last summer has this chi value within 0.1 - so within ~0.04 should be fine esp since it is just one data point. atp doing more calibration on this point will take a significant amount of time and I want to go to dinner it is pretty late already -> if time before open house tomorrow morning, recalib this point... good enough for now

 decided to take more data points for the gammas at smaller chis since the amount I was taking was producing too small gammas even after 3 iterations of gamma find->ratio tune->check gamma and repeat


 The reason the other plots agree more with theory than adj theory is bc the slight phase variation is bringing the resulting W_5 witness value down very very slightly... if you correct to match the actual produced gamma value (using gamma calculation method), I can almost guarantee this would NOT occur....

CHI->gamma_check value->tomo file's gamma (from processing data)
- 18->-931->-.75
- 36->.998->-.89
- 54->.983->-.94
- 72->.991->-1.13
- 90->.985->-1.04
->NOTE: maybe try retaking the chi 18 data point to see if you can improve it, but other than that I think it's aboutdone


# 07/17/2025
MP: Ria

I confirmed no pure states are witnessed by the W5 t1s... just to get more intuition for how they work, I generated a bunch of data using only the T1 witness values and looked at it.

Ultimately, I found that:
- all states with just H&V or just D&A or just R&L are witnessed by W3
- all states with HV & DA are witnessed by W5_t3
- all states with HV & RL are witnessed by W5_t2
- all states with DA & RL are witnessed by W5_t1
So, there are pure states witnessed by triplet one, they just don't have H/V on alice's side and are thus not states we can make.

Other info for chi18 data point to retake it:
 18: -131.29599484 & -21.986341777600742

- HH Counts: 1336.1+/-3.5
- VV Counts: 1315+/-6
- VH Counts: 12.3+/-0.7
- HV Counts: 14.1+/-0.5
- AD Counts: 25.5+/-0.9
- DA Counts: 28.2+/-0.8
- DD Counts: 1305+/-10
- AA Counts: 1299+/-5