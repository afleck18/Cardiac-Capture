# Cardiac-Capture *(Purdue BME Senior Design Project, Fall 2022)*
## Executive Summary
	Software Cardiac Capture aims to address problems of low readability and poor transferability of paper ECG scans. Currently, fax is used for 90% of transmitted medical communication which has led to a median accuracy of 75% when diagnosing patients based on ECG scans. Existing methods to solve this problem fall short; the three best attempts are an Instructional ECG App, hand calipers, and digitization of the ECG signal. Each is missing a key feature whether it’s transferability, portability, high image quality, or high accuracy. Cardiac Capture, however, addresses all of these key clinical needs and aims to fulfill this opportunity space. By filling this gap, a large market share is available, consisting of cardiac specialists, other medical professionals, and medical students/professors. 
 
	Our solution, Cardiac Capture, functions as an iOS smartphone application that allows the user to scan or upload a paper or digital ECG, implements an image processing algorithm that filters and cleans the image, allows the user to optimize their filtering options, and then allows for the secure storage and transfer of this ECG data. The filtering algorithm implements a series of image processing methods, to calibrate the image, remove the original grid, filter unnecessary noise, and calculate ECG physical characteristics. These algorithms also provide the user with an automatic measurement for the RR interval. The user of the app is also able to resize the ECG and measure specific sections to provide a proper diagnosis. These measurements can be inputted and stored in the patient’s files. Each user will have their own profile where they can access all of their patients and transfer files to another user. 

	The file’s database includes information about patient demographics, past measurements, and past ECG scans. This innovates on past designs by having an extensive patient database, being extremely portable, and having much better image quality. These innovative conclusions were drawn from our testing which involved verifying the algorithms used and comparing the actual values with expected ones. 

	The artifact removal verification and validation was done by using boundary recall values of our algorithm and comparing it to another, commonly used algorithm. The artifact removal didn’t work as well as the group had hoped. It was not significantly different from that of the common algorithm, which didn’t increase image quality in much. Interval calculation testing compared the computed RR intervals with those that were actually measured. The percent differences were fairly small (1.88%, 1.72%, 1.67%) except for one example (40.04%). Finally, the usability of the user interface was tested through a survey and were rated aesthetically pleasing by 87.2% of people. Moving forward, our algorithms need to be refined and integrated through Swift into the smartphone application. Once this is done, the app can be added to the iOS App Store through the Apple Developer Program. Finally, the app can be commercially marketed and utilized by its intended market.


## Environment Setup:
| Packages  | Version |
| :---         |     :---:      |
| altgraph | 0.17.3 |
| numpy | 1.24.1 |
| opencv-python | 4.7.0.68 |
| pefile | 2023.2.7 |
| Pillow | 9.4.0 |
| pyinstaller | 5.13.0 |
| pyinstaller-hooks-contrib | 2023.5 |
| pywin32-ctypess | 0.2.2 |

