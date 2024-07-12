# Detection-of-Age-and-Gender-of-people-in-a-Meeting-Room
This project detects the number of female and male in a meeting room along with their age. (Along with it has some constraints)

## To simply run this model
1. Clone this repository
2. Download the file [Age_Gender_Detection.keras](https://drive.google.com/file/d/1Relr0YTmSFWCbx_KHPCXtv_P4aspTaKD/view?usp=sharing) from this Google Drive.
3. Run this [gui.py](https://github.com/Me20b077/Detection-of-Age-and-Gender-of-people-in-a-Meeting-Room/blob/main/gui.py) file
4. Upload any image of a meeting room that contains group of people.
5. Click on Detect button
6. View Results
7. After running this file, an image with detected people and their ages will be created in your environment. Just see that image for verification.


## This Project has certain constraints
1. If a person wears white shirt, it detect his/her age as 23 irrespective of their gender and age.
2. If the total no of people is less than 2 and a person wears black shirt, it predicts him/her as a child irrespective of their gender and age.
