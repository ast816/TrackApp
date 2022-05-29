<h1 align="center">TrackApp
  <img src="https://is2-ssl.mzstatic.com/image/thumb/Purple128/v4/4a/8a/1b/4a8a1be9-e669-3fa4-05cf-18b8862a35ed/source/256x256bb.jpg" alt="Logo" width="25" height="25">
</h1>
<p align="center"> 
 <a target="_blank" href="https://youtu.be/rBkdrC8eD1Y">Video Demo</a>
</p>
    
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#brief-about-the-project">Brief About The Project</a>
      <ul>
        <li><a href="#salient-features">Salient Features</a></li>
        <li><a href="#Applications-of-This-Facial-Recognition-System">Applications of This Facial Recognition System</a></li>
        <li><a href="#Techstacks">Techstacks</a></li>
      </ul>
    </li>
    <li>
      <a href="#agile-methodology">Agile Methodology</a>
      <ul>
        <li><a href="#what-is-agile-methodology">What is Agile Methodology</a></li>
        <li><a href="#how-agile-methodology-helped-me">How Agile Methodology helped me</a></li>
      </ul>
    </li>
    <li>
      <a href="#Setup-and-getting-started">Setup and Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#navigating-through-the-app">Navigating through the App</a></li><ul>
        <li><a href="#Home">Home</a></li>
        <li><a href="#Hello-New-User">Hello New USer</a></li>
        <li><a href="#Test-Model">Test Model</a></li>
      </ul>
    <li><a href="#resources-used">Resources Used</a></li>
  </ol>
</details>
<!-- ABOUT THE PROJECT -->

## Brief About The Project
* Trackup Web Application is a project built during Microsoft Engage 202 program. 
* It is a Web Application based on Facial Detection for easy Database Navigation 
* It displays details of user on the basis of face recognition

### Salient Features
* Home page has User Manual and brief description of project.
* Hello New User page allows new users to enter their information which gets stored in a remote database.
* Test Model Page allows users to upload their recent photograph.
* They can then proceed to test the model by selecting camera device and check if the displayed details match input data.

### Applications of This Facial Recognition System
* Detecting Criminals and Missing people
* Identity Verifications
* Medical and Covid based applications
* The main idea is to use face as an Adhaar to build robust and comprehensive Databases for easier Navigation

### Techstacks

* [
ML Model-Python and various Modules-face_recognition,opencv-python,Pillow and others](https://img.shields.io/badge/ML%20Model-Python%20and%20various%20Modules--face__recognition%2Copencv--python%2CPillow%20and%20others-blue)
* [Frontend and backend -Streamlit -Open Source Framework to create Webapps](https://img.shields.io/badge/Front--end--and--Back--end-Streamlit%20--Open%20Source%20Framework%20to%20create%20Webapps-blue)
* [Database-Microsoft Azure SQL Server](https://img.shields.io/badge/Database-Microsoft%20Azure%20SQL%20Server-blue)

<!-- AGILE METHODOLOGY -->
## Agile Methodology
### What is Agile Methodology
* Agile software development is an umbrella term for a set of frameworks and practices based on the values and principles expressed in the Manifesto for Agile Software Development and the 12 Principles behind it. When you approach software development in a particular manner, it’s generally good to live by these values and principles and use them to help figure out the right things to do given your particular context.
* It promotes teamwork, flexible procedures, and sle-organizing teams.
### How Agile Methodology Helped me
SCRUM is a subset of Agile, a framework for developing software. It is now being successfully implemented in other business domains.Various concepts of iterative processes, continuous improvement, empowerment, and collaboration can be applied to Individual Projects as well.
* Sprint 1 (May 6): Planning and Research - Researching about the Project options, Exploring various applications of Facial Recognition and contemplating about solving a real-life problem with the available technologies. I also learnt some basic python in this sprint.
* Sprint 2 (May 16): Designing and Development- Started the development process by taking help from YouTube tutorials. Built the ML model using face_recognition library as it has high accuracy and trains with one single image. I also made an Azure SQL database and rendered 32MB storage to it. Then I installed SSMS(SQL Server Management Studio) to create table of sample details and connected my Azure database to my Model
* Sprint 3 (May 24):  Debugging and adding additional features - My detailed research made it easy for me to decide how to Implement backend and frontend of my Web App. I used streamlit due to its ease in building faster Webapps providing interface for both backend and frontend in pure python. I made the basic layout and attatched my Azure database to the web application and the Image uploading function with my training path.
<!-- INSTALLATIONS -->
## Setup and Getting Started
To install and run the project on your local system, following are the requirements:

### Prerequisites
* 1)[Download Python 3.10.4](https://www.python.org/ftp/python/3.10.4/python-3.10.4-amd64.exe)
* 2)[Visual Studio Code or any other IDE](https://code.visualstudio.com/)
* 3)Ensure you have [Visual Studio Community 2019](https://visualstudio.microsoft.com/downloads/) with Desktop Development in C++
* 4)Ensure you have the following extensions
* a) Python v2022.6.2 IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), Jupyter Notebooks, code formatt
* b) CodeRunner v0.11.7 Run C, C++, Java, JS, PHP, Python, Perl, Ruby, Go, Lua, Groovy, PowerShell, CMD, BASH, F#, C
* c) Azure Account v0.10.1 A common Sign In and Subscription management extension for VS Code.
* d) SQL Server (mssql) v1.14.2 Develop Microsoft SQL Server, Azure SQL Database and SQL Data Warehouse everywhe


### Installation

* Install [Cmake](https://cmake.org/download/) and go to environment variables and add it to the path
* Clone the Project and go to the project folder to run the following command to install all dependencies
* Run
```
pip install -r /path/to/requirements.txt
```

* To run the streamlit application just go to any app file(change directory to the required folder) and run
```
streamlit run app.py
```

<!-- APP TUTORIAL-->
## Navigating Through The App
### Home
Home Page appears as soon as you launch the web-application. You can check out the project description and User Manual here
<img src="Face-Recognition-Attendance-Projects/apps/step 1.jpg" alt="Home screen" width="700"/>
### Hello New User
* New users can enter their details and click on the submit button 
<img src="Face-Recognition-Attendance-Projects/apps/step 2.jpg" alt="Hello New User screen" width="700"/>
* They shall be greeted with balloons when submission is successful.
<img src="Face-Recognition-Attendance-Projects/apps/step 3.jpg" alt="Hello New User screen" width="700"/>
* All the users details get stored in the azure SQL Database
### Test Model
* Users can now upload their recent photograph here. Once saved Successfully they shall be prompted about the same
<img src="Face-Recognition-Attendance-Projects/apps/step 4.jpg" width="700"/>
* They can then choose to turn on their webcam locally and their face gets recognised
<img src="Face-Recognition-Attendance-Projects/apps/step 5.jpg" alt="face recognised" width="700"/>
* Details of the recognised person are displayed, If another person is in front of webcam then their details are printed if they are saved in database
<img src="Face-Recognition-Attendance-Projects/apps/step 6.jpg" alt="Details printed" width="700"/>
<!-- ACKNOWLEDGEMENTS -->
## Resources Used
* 1)[Streamlit](https://streamlit.io/)
* 2)[Face_recognition](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
* 3)[Face recognition based Streamlit Application](https://www.youtube.com/watch?v=p80IQSNf7LU&t=1837s)

