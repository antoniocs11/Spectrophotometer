# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:18:09 2023

@author: usuario
"""
# Libraries needed for the program

import cv2
import os
import datetime as dt
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt
import tensorflow
import joblib
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import figure
from tkinter import *
from pathlib import Path
from tkinter import filedialog
from PIL import Image
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy import signal
import csv
from tkinter import messagebox
import tkinter as tk
import matplotlib.colors as mcolors


#Necessary code to create the interface through library "Tkinter"

raiz= Tk()
raiz.title("Spectrophotometer")

#Function to extract path to the directory where the current script is located. It will be used later

def directory_path():
    # Gets the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # Gets the directory of the current script
    directory_script = os.path.dirname(script_path)

    # Modifies the directory path so that it can be read by python at a later time
    directory_process = directory_script.replace("\\" , "\\\\")

    return directory_process

#----------------------------Spectrophotometer Functions-------------------------------------------


# The variables named with "global" are to be stored outside this function and can be used in any area of the code and program.

# Function "examine" is associated with button 5, which is responsible for loading the file to be studied. This function stores the file path and displays it on the interface. It also resets all the text boxes of the interface and the graph that appears on the right.

def examine():
    global name1 #variable to save path of  selected video or image 
    global example1 #variable to show this path in the screen of the interface in the text box 3.
    name1 = filedialog.askopenfilename()
    example1.set(name1)
    
    #All text boxes of the interface are shown in blank, except text box 3 which is showing selected path. 
    wave.set("") #variable associated with text box 10
    wave1.set("") #variable associated with text box 5
    tframes.set("") #variable associated with text box 9
    tframes1.set("") #variable associated with text box 4
    process.set("") #variable associated with text box 6
    saved.set("") #variable associated with text box 8
    
    #Graph of the interface is initially shown empty, without any information
    fig, axs =plt.subplots(1,1,dpi=80, figsize=(7,5), sharey=True)
    fig.suptitle('')
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)

#Function "timef" is associated with button 16, which is responsible to saved time, introduce for the user (in the text box 4), in the system and be able to show in the text box 9
def timef():
    global time #variable that save the time introduce for the user in the text box 4
    time=float(text_box4.get()) #The time added by the user is saved in the variable "time"
    tframes.set(time) #The time introduced by the user appears in the text box 9
    time=float(text_box9.get())
    tframes1.set("") #Text box 4 is changed to blank

#Function "landa" is associated with button 17, which is responsible to save wavelength, introduced by the user(in the text box5), in the system and be able to show it in the text box 10
def landa():
    global lamda #variable that save the wavelength introduce by the user in the text box 5
    lamda=int(text_box5.get())#The wavelength add by the user is saved in the variable "lamda"
    wave.set(lamda)#The wavelength introduced by the user appears in the text box 10
    lamda=int(text_box10.get())
    wave1.set("") #Text box 5 is changed to blank

#Function "calculate" is associated with button 8, which is responsible to show the first frame of the video or the first image of the folder selected in another window. In this window, the user could select the desired area to study.
def calculate():
    #EXTRACTION OF VIDEO FRAMES AND LAB CORRESPONDING TO THE SELECTED AREA
    global n #variable that will save the number of obtained frames of the video selected or the number of images that appeared in the folder selected by the user. 
    global s #variable to identify if the selected process is "Spectrum kinetics" or "Single Spectrum Kinetics"
    global Labim #variable to save Lab coordinates of the frames or images studied during the process
    global nx # number of different wavelengths in each spectrum
    global Labim_norm2 #variable to save Lab coordinates from the variable "Labim", but it will be normalized between 0 and 1 when introduced in the neural network 
    global sol_n_n # variable to save neural network results.
        
    s=0 #reset variable "s"
    
    if option1==1: #User has selected "Video" option
        
        # Read the video from specified path saved in variable "name1"
        
        vid = cv2.VideoCapture(name1)
        
        #A new folder is created (in the same folder where the script is saved) to save all frames that will be extracted from the video
    
        try:
    
            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
    
        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')
    
        
        currentframe = 0
        frame_count = 0
        fps = vid.get(cv2.CAP_PROP_FPS) #variable to extract the number of frames per second of the video
        interval_frames = int(fps * time) #number of frames specifying how often a frame should be extracted
        n=0  #reset of the variable "n"
        while (True):
    
            # reading from frame
            success, frame = vid.read()
                        
            if success:
                                
                # continue creating images until video remains
                if frame_count % interval_frames == 0:
                    
                    n=n+1
                    name = './data/frame' + str(currentframe) + '.jpg'
                    print('Creating...' + name)
                    
    
                    # writing the extracted images
                    cv2.imwrite(name, frame)
    
                    # increasing counter so that it will show how many frames are created
                    currentframe += 1
                    
            else:
                
                
                break
            frame_count += 1
    
        # Release all space and windows once done
        vid.release()
        cv2.destroyAllWindows()
    
        #SELECT PART OF THE IMAGE TO EVALUATE
            
        #Function for the selection of the area of video extracted frames to process
        def drawing(event,x,y,flags,param):
            global pixelx1, pixelx2, pixely1, pixely2   #variables that will save pixels of the selected area by the user
            
            
            if event == cv2.EVENT_LBUTTONDOWN: # By clicking the left mouse button, the upper left corner of the area to be studied is selected.
                
                #Pixels in X axis and Y axis are saved in variable "pixelx1" and "pixely1"
                pixelx1=x
                pixely1=y
                #The selected pixels are displayed on the screen
                print ('pixel x1=',x) 
                print ('pixel y1=',y)
            if event == cv2.EVENT_RBUTTONDOWN: # By clicking the right mouse button, the bottom right corner of the area to be studied is selected.
                
                #Pixels in X axis and Y axis are saved in variable "pixelx2" and "pixely2"
                pixelx2=x
                pixely2=y
                #The selected pixels are displayed on the screen
                print ('pixel x2=',x)
                print ('pixel y2=',y)
                cv2.rectangle(image,(pixelx1,pixely1),(pixelx2,pixely2),(255,0,0),1)
        	
        #Code to show the first frame obtained of the video in another window and code necessary to call the previous function
        image = cv2.imread('./data/frame0.jpg')
        image_to_study = image.copy() 
    
        cv2.namedWindow('image_to_study')
        cv2.setMouseCallback('image_to_study',drawing)
    
    
        while True:
        	cv2.imshow('image_to_study',image)
        	
        	if cv2.waitKey(1) & 0xFF == 27:
        		break
    
        cv2.destroyAllWindows()
        
        x_axis=abs(pixelx2-pixelx1) #Variable to save the number of pixels in x axis from the area selected by the user
        y_axis=abs(pixely2-pixely1) #Variable to save the number of pixels in y axis from the area selected by the user
        #It is shown the numbers of pixels in Y and X axis from the area selected by the user
        print ('x_axis=',x_axis)
        print ('y_axis=',y_axis)
        
        p=x_axis*y_axis #variable to save the number of total pixels from the area selected by the user
        Labim=np.zeros((n,3)) 
        
        nx=61 #The number of different wavelengths in each spectrum will be 61, one per 5 nm (400-700 nm)
        
        #The function "directory_path" is called to extract directory path in which the script has been saved
        if __name__ == "__main__":
            
            Process_path = directory_path()
        
        #Necessary loop to extract Lab coordinates from each pixel of the selected area. Then, average value of each coordinate is obtained
        for Counter in range(n): 
            image1=Image.open(Process_path + '\\data\\frame' + str(Counter) + '.jpg') #frame by frame is studied, focusing in the area selected by the user
            
            Lab3=np.zeros((p,3)) # variable to save Lab coordinates of each pixel of the area selected
            c=0
            #Necessary loop to obtain Lab coordinates from each pixel in each frame 
            for i in range(y_axis):
                for j in range(x_axis):
                    r, g, b = image1.getpixel((j+pixelx1,i+pixely1))
                    im1 = sRGBColor(r/255, g/255, b/255)
                    im2= convert_color(im1, LabColor) 
                    Lab3[c,0]=im2.lab_l+5
                    Lab3[c,1]=im2.lab_a
                    Lab3[c,2]=im2.lab_b
                    c=c+1
                    
            L=0
            a=0
            b=0
            for i in range(p):
                L=L+Lab3[i,0]
                a=a+Lab3[i,1]
                b=b+Lab3[i,2]
            
            Labim[Counter,:] = [L/p,a/p,b/p] #Average Lab coordinates of the selected area are saved by the user
            
    
        #DATA NORMALIZATION
        
        Labim_norm1=np.zeros((n*nx,4))
        Labim_norm2=np.zeros((n*nx,4))
        y=0
        for i in range (n):
          for j in range (nx):
            Labim_norm1[y,0]=Labim[i,0]
            Labim_norm1[y,1]=Labim[i,1]
            Labim_norm1[y,2]=Labim[i,2]
            Labim_norm1[y,3]=400+j*5
            Labim_norm2[y,0]=Labim_norm1[y,0]/100
            Labim_norm2[y,1]=(Labim_norm1[y,1]+100)/200
            Labim_norm2[y,2]=(Labim_norm1[y,2]+100)/200
            
            if Labim_norm1[y,3] == 400:
              Labim_norm2[y,3]=0
            else:
              Labim_norm2[y,3]=(Labim_norm1[y,3]-400)/300
    
            y=y+1
            
        
        import tensorflow
        import joblib
        from keras.models import load_model
        
        sol_n_n = load_model('Neural_network.h5') # the file containing all the neural network data is loaded. This file should be saved in the same folder as this script
    
        process.set("Process completed") #The process has been finished and the text box 6 shows it
    
    if option1==2: #User has selected "Image sequence" option
               
        #code lines to extract the name of folder in which the images are
        f=0
        variable=-1
        
        while (f<2):
            
            if name1[variable]=='/':
                
                variable=variable+1
                name6=name1[variable:]
                f=f+1
                variable=variable-2
            else:
                variable=variable-1
        
        f=0
        variable=1
        while (f==0):
            
            if name6[variable]=='/':
                
                name7=name6[:variable]
                f=1
            else:
                variable=variable+1
        
        def drawing(event,x,y,flags,param): #The same function created in video section to obtain the selected area by the user
            global pixelx1, pixelx2, pixely1, pixely2
        	   
            if event == cv2.EVENT_LBUTTONDOWN:
                pixelx1=x
                pixely1=y
                print ('pixel x1=',x)
                print ('pixel y1=',y)
            if event == cv2.EVENT_RBUTTONDOWN:
                pixelx2=x
                pixely2=y
                print ('pixel x2=',x)
                print ('pixel y2=',y)
                cv2.rectangle(image,(pixelx1,pixely1),(pixelx2,pixely2),(255,0,0),1)
        	
        #code to show the image selected by the user in another window and code necessary to call the anterior function
        image = cv2.imread('./'+str(name6))
        image_to_study = image.copy() 
    
        cv2.namedWindow('image_to_study')
        cv2.setMouseCallback('image_to_study',drawing)
        
        while True:
        	cv2.imshow('image_to_study',image)
        	
        	if cv2.waitKey(1) & 0xFF == 27:
        		break
    
        cv2.destroyAllWindows()
        
        #code lines to extract the images folder path in variable "name8"
        
        f=0
        variable=-1
        
        while (f==0):
            
            if name1[variable]=='/':
                variable=variable+1
                name8=name1[:variable]
                f=1
            else:
                variable=variable-1
        
        folder = name8
    
        # Gets the list of files in the folder
        files = os.listdir(folder)
    
        # Filter the list to include only files (not directories).
        files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    
        # Count the number of files
        n = len(files)
        
        x_axis=abs(pixelx2-pixelx1) #variable to save the number of pixels in x axis from the area selected by the user
        y_axis=abs(pixely2-pixely1)#variable to save the number of pixels in y axis from the area selected by the user
        #The numbers of pixels in Y and X axis corresponding to the area selected by the user are shown
        print ('x_axis=',x_axis)
        print ('y_axis=',y_axis)
    
        # Extract images and convert into color coordinates
        
        p=x_axis*y_axis #variable to save the number of total pixels from the area selected by the user
        Labim=np.zeros((n,3)) 
        
        nx=61 #The number of different wavelengths in each spectrum will be 61, one per 5 nm (400-700 nm)
        counter=0
        file=0
        global File_path2
        for file in os.listdir(folder): #Necessary loop to open each image and extract Lab coordinates from the selected area 
            # Complete file path
            File_path2 = os.path.join(folder, file)
        
            # Opens each image in the selected folder
            if os.path.isfile(File_path2) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image1=Image.open(File_path2)
                       
            Lab3=np.zeros((p,3)) # variable to save  Lab coordinates of each pixel of the area selected
            c=0
            #Necessary loop to obtain Lab coordinates from each pixel in each image
            for i in range(y_axis):
                for j in range(x_axis):
                    r, g, b = image1.getpixel((j+pixelx1,i+pixely1))
                    im1 = sRGBColor(r/255, g/255, b/255)
                    im2= convert_color(im1, LabColor) 
                    Lab3[c,0]=im2.lab_l
                    Lab3[c,1]=im2.lab_a
                    Lab3[c,2]=im2.lab_b
                    c=c+1
            
            L=0
            a=0
            b=0
            for i in range(p):
                L=L+Lab3[i,0]
                a=a+Lab3[i,1]
                b=b+Lab3[i,2]
            
            Labim[counter,:] = [L/p,a/p,b/p] #Average Lab coordinates of the user selected area are saved
            counter=counter+1
                
        #DATA NORMALIZATION
        
        Labim_norm1=np.zeros((n*nx,4))
        Labim_norm2=np.zeros((n*nx,4))
        y=0
        for i in range (n):
          for j in range (nx):
            Labim_norm1[y,0]=Labim[i,0]
            Labim_norm1[y,1]=Labim[i,1]
            Labim_norm1[y,2]=Labim[i,2]
            Labim_norm1[y,3]=400+j*5
            Labim_norm2[y,0]=Labim_norm1[y,0]/100
            Labim_norm2[y,1]=(Labim_norm1[y,1]+100)/200
            Labim_norm2[y,2]=(Labim_norm1[y,2]+100)/200
            
            if Labim_norm1[y,3] == 400:
              Labim_norm2[y,3]=0
            else:
              Labim_norm2[y,3]=(Labim_norm1[y,3]-400)/300
    
            y=y+1
        
        import tensorflow
        import joblib    
        from keras.models import load_model
        
        sol_n_n = load_model('Neural_network.h5') #the file containing all the neural network data is loaded. This file should be saved in the same folder as this script
        
        process.set("Process completed") #The process has been finished and the text box 6 shows it
        
#Function "SK" is associated with button 9. It will be executed when the user decide to select "spectrum kinetics" process
def SK():
    
    global contrast2 #variable to save maximum contrast from the obtained spectra 
    global wlength1 #variable to save the wavelength in which contrast is maximum
    global wlength #variable to create the entire wavelength range between 400 and 700 nm
    global s
    global spectrum_pred1 #variable to save spectrum results from the neural network
       
    s=1 #It is associated with number 1 in "PDF" function
            
    spectrum_pred=sol_n_n.predict(Labim_norm2) #Spectrum result from neural network, in normalized (0 to 1) values
    spectrum_pred1=np.zeros((n,nx))
    y=0
    #Necessary loop to undo normalization
    for i in range (n):
      for j in range (nx):
        spectrum_pred1[i,j]=spectrum_pred[y,0]
        y=y+1
    
    #Code lines to extract maximum contrast and the corresponding wavelength.
    contrast=0
    contrast1=0
    for j in range (nx):
      max=0
      min=1
      for i in range (n):
        if spectrum_pred1[i,j]>max:
          max=spectrum_pred1[i,j]
        if spectrum_pred1[i,j]<min:
          min=spectrum_pred1[i,j]
      
      contrast=max-min
      if contrast>contrast1:
        contrast1=contrast
        wlength1=j*5+380

    contrast1=contrast1*100
    contrast2 = round(contrast1, 2)
    
    #Creation of the matrix "wlength" (axis X to represent spectrum results graphically)
    wlength=np.zeros((61,1))
    z=0
    for i in range (61):
      wlength[i] = 400 + z
      z=z+5
    
    #Code lines to graphically represent spectrum results from the neural network in the interface.
    fig, axs =plt.subplots(1,1,dpi=80, figsize=(7,5), sharey=True)
    fig.suptitle('Spectrum Kinetics', size=20)
    
    for i in range (n):
        axs.plot(wlength,100*spectrum_pred1[i,:], color='y')
    
    
    
    axs.set_xlabel('Wavelength (nm)',size=12)
    axs.set_ylabel('Transmittance (%)',size=12)
    axs.set_ylim(0,100)
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)
    
#Function associated with button 10. It will be executed when the user decide to select "Single wavelength kinetics" process.
def SWK():
     
    global wlength3 #X axis to represent results from the neural network (represent time in seconds of each value)
    global spectrum_pred2 #variable to save spectrum results from the neural network. Now, for each image, the result is only for the selected wavelength
    global s
    
    lamda1 = (lamda-400)/300 #user-entered standardized wavelength 
    Labim_norm3=np.zeros((n,4)) #variable to store the results of the neural network when undoing normalization
    t=0
    s=2 #It is associated with number 2 to related in "PDF" function
    #Necessary loop to obtain only values from the wavelength selected by the user
    for i in range (n*nx):
        if Labim_norm2[i,3] == lamda1:
            Labim_norm3[t,0]=Labim_norm2[i,0]
            Labim_norm3[t,1]=Labim_norm2[i,1]
            Labim_norm3[t,2]=Labim_norm2[i,2]
            Labim_norm3[t,3]=Labim_norm2[i,3]
            t=t+1

    spectrum_pred2=sol_n_n.predict(Labim_norm3)#Results from the neural network with the values referred to the selected wavelength
    
    
    wlength3=np.zeros((n,1))
    #Creation of the matrix "wlength" (axis X to represent spectrum results graphically)
    for i in range (n):
        if i==0:
            wlength3[i] = 0
        else:
            wlength3[i]=wlength3[i-1]+time

#Code lines to represent graphically the spectrum results from the neural network in the interface.    
    fig, axs =plt.subplots(1,1,dpi=80, figsize=(7,5), sharey=True)
    fig.suptitle('Single Wavelength Kinetics', size=20)
    
    if option1==1:
        axs.plot(wlength3,100*spectrum_pred2, color='y')
    
    if option1==2:
        axs.plot(wlength3,100*spectrum_pred2, color='y')
    
        
    axs.set_xlabel('Time (seconds)',size=12)
    axs.set_ylabel('Transmittance (%)',size=12)
    axs.set_ylim(0,100)
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)
    
#Function associated with button 13. Clicking this button, the graph shown in the interface and related data are saved in three different file formats (PDF, .csv and Tiff) 
def PDF():
    
    global file #variable to store the name that the user has introduced in thext box 8
    file= text_box8.get()
    
    if s==1:# If the process selected is "Spectrum Kinetics"
        
        # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
        df1 = pd.DataFrame(wlength)
        df2 = pd.DataFrame(np.transpose(spectrum_pred1))

        # Combines the DataFrames horizontally into a single DataFrame.
        df_combined = pd.concat([df1, df2], axis=1)

        # Specify the name of the CSV file that you want to create.
        file_name = str(file)+".csv"

        # Save the combined DataFrame in a CSV file.
        df_combined.to_csv(file_name, index=False, header=False)
        
        with PdfPages(str(file)+'.pdf') as pdf: #Function to create PDF
            
            #Code lines to configure the graph that will appear in the PDF file
            txt="Spectrum Kinetics"
            firstPage=plt.figure(figsize=(10,5))
            firstPage.clf()
            firstPage.text(0.5,0.9,txt,transform=firstPage.transFigure,size=24,ha='center')
            
            for i in range (n):
              plt.plot(wlength,100*spectrum_pred1[i,:], linestyle = "-", color='y')
              plt.ylim(0,100)
           
            plt.text(550, 95, 'Maximum contrast =' + str(contrast2) + '%', fontsize=10, color='black')
            plt.text(550, 90, 'Wavelength (nm) = ' + str(wlength1), fontsize=10, color='black')
            plt.xlabel('Wavelength (nm)',size=18)
            plt.xscale
            plt.ylabel('Transmittance (%)',size=18)
            plt.savefig(file, format='tif')#To save in tiff format

            pdf.savefig()
            plt.close()
            
        saved.set("Saved")
        
    if s==2: # If the process selected is "Single Wavelength Kinetics"
        
        # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are necessary to create the "csv" file
        df1 = pd.DataFrame(wlength3)
        df2 = pd.DataFrame(spectrum_pred2)

        # Combines the DataFrames horizontally into a single DataFrame.
        df_combined = pd.concat([df1, df2], axis=1)

        # Specify the name of the CSV file that you want to create.
        file_name = str(file)+".csv"

        # Save the combined DataFrame in a CSV file.
        df_combined.to_csv(file_name, index=False, header=False)
        
        with PdfPages(str(file)+'.pdf') as pdf: #Function to create PDF
            
            #Code lines to configure the graph that will appear in the PDF
            txt="Single Wavelength Kinetics"
            firstPage=plt.figure(figsize=(10,5))
            firstPage.clf()
            firstPage.text(0.5,0.9,txt,transform=firstPage.transFigure,size=24,ha='center')
            
            if option1==1:
  
                for i in range (n):
                  plt.plot(wlength3,100*spectrum_pred2, linestyle = "-", color='yellow')
                  plt.ylim(0,100)
                  
            if option1==2:
                
                for i in range (n):
                  plt.plot(wlength3,100*spectrum_pred2, linestyle = "-", color='yellow')
                  plt.ylim(0,100)
                
            plt.text(int(n/time)*0.75, 0.90, 'Wavelength = ' + str(lamda)+'nm', fontsize=10, color='black')
            plt.xlabel('Time (seconds)',size=18)
            plt.xscale
            plt.ylabel('Transmittance (%)',size=18)
            plt.savefig(file, format='tif') #To save in tiff format

            pdf.savefig()
            plt.close()
        
        saved.set("Saved")
        
#Function associated with button 14. This function reset all text boxes of the interface
def clear():
    
    global time
    global lamda
    #"time" and "lamda" variables are reset to 0
    time=0
    lamda=0
    
    #All text boxes of the interface are reset
    wave.set("")
    wave1.set("")
    tframes.set("")
    tframes1.set("")
    process.set("")
    saved.set("")
    example1.set("")
    
    #Graph of the interface is reset
    fig, axs =plt.subplots(1,1,dpi=80, figsize=(7,5), sharey=True)
    fig.suptitle('')
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)

    
#All next functions are associated with info buttons 
def show_explanation():
    explanation = """
    Type of files that can be processed by this spectrophotometer are videos or a sequence of images. Please choose one of them.    
    """
    messagebox.showinfo("File Type", explanation)

def show_explanation1():
    explanation = """
    Clicking the button “Upload file”, the user can browse and upload the video or the sequence of images to study.  
-	If a video is selected, please choose this file from any folder. 
-	If a sequence of images is selected, please follow the next steps to operate the system properly:
1.	Please, obtain the images at a regular interval (every one second for example).
2.	Save the set of images in a single folder. Besides, save this folder in the same folder in which the script of this software is running. 
3.	Select the first image to be processed, the system will process from this file on.  
    """
    messagebox.showinfo("Step 1", explanation)

def show_explanation2():
    explanation = """
    If video option is selected, please indicate the desired interval (in seconds) for data processing. 
    If Image sequence is selected, please indicate the interval (in seconds) with which the images were taken. 
    In both cases, please indicate this interval in the first text box and then click in “Enter data” button. Automatically, this data will be saved in the system.
    """
    messagebox.showinfo("Step 2", explanation)

def show_explanation3():
    explanation = """
    To select the area of the images or video to study, please follow the next steps:
1.	First, press the button “Select area”.
2.	Please, wait until an image of the selected file appears in a pop-up window.
3.	Then, with the left mouse button, please select the upper left corner of the desired area of study. 
4.	With the right mouse button, please select the lower right corner, forming a rectangle corresponding to the area to be studied.
5.	Press "esc" key. 
6.	The process could take some time, mainly if the set of images is big or the video length is large. Please, wait until the process finishes. When the task is done, the next message will appear: “Process completed”
    """
    messagebox.showinfo("Step 3", explanation)

def show_explanation4():
    explanation = """
    If the data processing method to be selected is Single Wavelength Kinetics, then, please indicate this wavelength (in nanometers). Please, introduce this value in the first text box and then click in “Enter data”. Automatically, this data will be saved in the system.
    On the contrary, if the data processing method to be selected is Spectrum Kinetics, please leave this space in blank.  
    """
    messagebox.showinfo("Step 4", explanation)

def show_explanation5():
    explanation = """
    Two options can be selected by clicking the corresponding button:
1.	Spectrum kinetics: Transmittance spectra in visible range of the selected area will be shown in one graph. If a video file has been processed, a set of spectra (one spectra every x seconds, being x the previously defined “data acquisition interval”) will be displayed in the graph. If a sequence of images has been processed, a set of spectra (one spectrum per image) will be displayed in the graph.
2.	Single wavelength Kinetics: The evolution of the transmittance vs. time of the previously selected area and wavelength will be plotted on a graph. The number of data points will correspond with the previously defined interval.
    """
    messagebox.showinfo("Step 5", explanation)

def show_explanation6():
    explanation = """
    For properly saving the obtained data, please follow next steps:
1.	Introduce the file name to save the data. 
2.	Then, press the button “Save file”.  Data will be saved in the same folder in which the software script is. Three different format files with the same information will be created and saved: “PDF”, “TIFF” and “csv”. 
Clicking the “Clear” button will delete all the information in the screen allowing to continue processing new data files. 

    """
    messagebox.showinfo("Step 6", explanation)

#function associated with button "drop_down_menu"

def select_option(option):#This function let the user select video or image sequence
    global option1# Variable to save the selected option by the user
    k.config(text="Selected option: " + option)
    if option=="Video":
        option1=1
    
    if option=="Image_sequence":
        option1=2

options = ["Video", "Image_sequence"]


#------------------------interface configuration in tkinter----------------------------------------

padx=5 #x-axis separation of buttons and textboxes (mm)
pady=5 #y-axis separation of buttons and textboxes (mm)
background="yellowgreen" # Colour background of the interface

#configuration of the interface
frame2=Frame()
frame2.pack(fill="both",expand="True")
frame2.config(bg=background)
frame2.config(bd=35) #frame border width
frame2.config(relief="flat")  #for border, border type

#Configuration of the text that appears in the interface
Label(frame2, text="File Type", fg="black",bg=background,font=("arial",14)).grid(row=0, column=0, sticky="w", padx=10, pady=10)
Label(frame2, text="1. Browse: upload video or image sequence.", fg="black",bg=background,font=("arial",14)).grid(row=1, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="2. Select data acquisition interval.", fg="black",bg=background,font=("arial",14)).grid(row=3, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="3. Select area to be processed.", fg="black",bg=background,font=("arial",14)).grid(row=5, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="4. Select wavelength to study (optional).", fg="black",bg=background,font=("arial",14)).grid(row=7, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="5. Select a process.", fg="black",bg=background,font=("arial",14)).grid(row=9, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="6. Save file as.", fg="black",bg=background,font=("arial",14)).grid(row=11, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="Time (seconds)", fg="black",bg=background,font=("arial",14)).grid(row=4, column=0,padx=padx, pady=pady)
Label(frame2, text="Wavelength (nm)", fg="black",bg=background,font=("arial",14)).grid(row=8, column=0,padx=padx, pady=pady)

#Configuration of the arrow used in the interface in several buttons
arrow = "\u2192"

text_with_arrow = f"Enter data {arrow}"
#Configuration of all buttons that appear in the interface

button5=Button(frame2, text="Upload file", fg="white", bg="black",font=("arial",14), command=examine)
button5.grid(row=2,column=0,padx=padx, pady=pady)

button8=Button(frame2, text="Select Area", fg="white", bg="black",font=("arial",14), command=calculate)
button8.grid(row=6,column=0,padx=padx, pady=pady)  

button9=Button(frame2, text="Spectrum Kinetics", fg="white", bg="black",font=("arial",14), command=SK)
button9.grid(row=10,column=0,padx=padx, pady=pady)

button10=Button(frame2, text="Single Wavelength Kinetics", fg="white", bg="black",font=("arial",14), command=SWK)
button10.grid(row=10,column=1,padx=padx, pady=pady)

button13=Button(frame2, text="Save file", fg="white", bg="black",font=("arial",14), command=PDF)
button13.grid(row=12,column=1,padx=padx, pady=pady)

button14=Button(frame2, text="Clear", fg="white", bg="black",font=("arial",14), command=clear)
button14.grid(row=12,column=3,padx=padx, pady=pady)

button16=Button(frame2, text=text_with_arrow, fg="white", bg="black",font=("arial",14),command=timef)
button16.grid(row=4,column=1,sticky="e",padx=padx, pady=pady) 

button17=Button(frame2, text=text_with_arrow, fg="white", bg="black",font=("arial",14),command=landa)
button17.grid(row=8,column=1,sticky="e",padx=padx, pady=pady) 

#Configuration of all information buttons

button_info = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info.grid(row=0,column=1,sticky="e",padx=0, pady=0)
button_info.bind("<Button-1>", lambda event: show_explanation())

button_info1 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info1.grid(row=1,column=1,sticky="e",padx=0, pady=0)
button_info1.bind("<Button-1>", lambda event: show_explanation1())

button_info2 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info2.grid(row=3,column=1,sticky="e",padx=0, pady=0)
button_info2.bind("<Button-1>", lambda event: show_explanation2())

button_info3 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info3.grid(row=5,column=1,sticky="e",padx=0, pady=0)
button_info3.bind("<Button-1>", lambda event: show_explanation3())

button_info4 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info4.grid(row=7,column=1,sticky="e",padx=0, pady=0)
button_info4.bind("<Button-1>", lambda event: show_explanation4())

button_info5 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info5.grid(row=9,column=1,sticky="e",padx=0, pady=0)
button_info5.bind("<Button-1>", lambda event: show_explanation5())

button_info6 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info6.grid(row=11,column=1,sticky="e",padx=0, pady=0)
button_info6.bind("<Button-1>", lambda event: show_explanation6())

#Configuration of the drop down menu in the interface
selected_option = tk.StringVar(frame2)
selected_option.set("")

drop_down_menu = tk.OptionMenu(frame2, selected_option, *options, command=select_option)
drop_down_menu.grid(row=0,column=1,sticky="w",padx=padx, pady=pady)

k = tk.Label(frame2)

#All variables, related with the text box of the interface, are named through StringVar variables

example1=StringVar()
process=StringVar()
saved=StringVar()
wave=StringVar()
tframes=StringVar()
wave1=StringVar()
tframes1=StringVar()

# Configuration of all the text boxes that appear in the interface
text_box3=Entry(frame2, textvariable=example1,font=("arial",14)) 
text_box3.grid(row=2, column=1,padx=padx, pady=pady)
text_box4=Entry(frame2, width=6,textvariable=tframes1,font=("arial",14))
text_box4.grid(row=4, column=1,sticky="w",padx=padx, pady=pady) 
text_box5=Entry(frame2,width=6,textvariable=wave1,font=("arial",14))
text_box5.grid(row=8, column=1,sticky="w",padx=padx, pady=pady)
text_box6=Entry(frame2, textvariable=process,font=("arial",14))
text_box6.grid(row=6, column=1,padx=padx, pady=pady) 
text_box8=Entry(frame2, textvariable=saved,font=("arial",14))
text_box8.grid(row=12, column=0,padx=padx, pady=pady)
text_box9=Entry(frame2, width=6,textvariable=tframes,font=("arial",14))
text_box9.grid(row=4, column=2,sticky="w",padx=50, pady=pady)
text_box10=Entry(frame2,width=6,textvariable=wave,font=("arial",14))
text_box10.grid(row=8, column=2,sticky="w",padx=50, pady=pady)

raiz.mainloop()

