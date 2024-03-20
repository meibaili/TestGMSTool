import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from ipywidgets import *
from select_GM import *
import threading
from IPython.display import display, HTML
import ipywidgets as widgets

# from scipy import integrate
# from scipy.optimize import curve_fit
# import matplotlib as plt
# import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

'''    
def setUpInputInterface():

    custom_css = """
    <style>
        .jupyter-widgets button {
            font-size: 20px;
        }
    </style>
    """
    display(HTML(custom_css))

    style = {'description_width': 'initial', 'font_size': '16px'}
    layout =  {'width': 'auto', 'max_width': '1500px'}
    label_style = '<style>.label {font-size: 16px; margin-right: 5px;}</style>'
    label_layout = Layout(width='400px')
    layout_button1 = {'width': 'auto', 'max_width': '1500px', 'margin': '25px 0 0 0', 'height': '35px'}
    layout_button2 = {'width': 'auto', 'max_width': '1500px', 'margin': '8px 0 0 0', 'height': '35px'}
    max_length = 70  # Max length of description to align all widgets

    def toggle_widgets(change):
        """
        Toggle the visibility of widgets based on the value of Elastic_field.
        """
        if change['new'] == 'Elastic (TargetSpectra_Elastic.csv)':
            numPulse_field.layout.visibility = 'visible'
            minPulseT_field.layout.visibility = 'visible'
            minDuration_field.layout.visibility = 'visible'
            ductility_checkList.layout.visibility = 'hidden'
            numPulse_field.layout.display = 'flex'
            minPulseT_field.layout.display = 'flex'
            minDuration_field.layout.display = 'flex'
            ductility_checkList.layout.display = 'none'
        else:
            numPulse_field.layout.visibility = 'hidden'
            minPulseT_field.layout.visibility = 'hidden'
            minDuration_field.layout.visibility = 'hidden'
            ductility_checkList.layout.visibility = 'visible'
            numPulse_field.layout.display = 'none'
            minPulseT_field.layout.display = 'none'
            minDuration_field.layout.display = 'none'
            ductility_checkList.layout.display = 'flex'
            
    def create_widget_with_label(label_text, widget):
        label_html = HTML(value=f"{label_style}<div class='label'>{label_text}</div>", layout=label_layout)
        #widget.layout = layout
        return HBox([label_html, widget])
        
    fileLoc_field = widgets.Text(
        value=f'{os.getcwd()}/Inputs',
        placeholder=f'{os.getcwd()}/Inputs',
        description='File path to elastic target folder:',
        disabled=True,
        style=style,
        layout = layout
    )
    # fileLoc_widget = create_widget_with_label("File path to elastic target folder:", fileLoc_field)

    Elastic_field = widgets.Dropdown(
        options=['Elastic (TargetSpectra_Elastic.csv)', 'Inelastic'],
        value='Elastic (TargetSpectra_Elastic.csv)',
        description='Consider only elastic target or inelastic target:',
        disabled=False,
        style=style,
        layout = layout
    )
    # Elastic_widget = create_widget_with_label("Consider only elastic target or inelastic target:", Elastic_field)
    Elastic_field.observe(toggle_widgets, names='value') # Observe changes in the value of Elastic_field
    numGM_field = widgets.Text(
        value='',
        placeholder='e.g., 11 | 11 ground motions to be selected in a suite',
        description='Number of records in a suite:', 
        disabled=False,
        style=style,
        layout = layout
    )
    magnitude_field = widgets.Text(
        value='',
        placeholder='e.g., 6-8 | range is 6 to 8',
        description='M:',
        disabled=False,
        style=style,
        layout = layout

    )
    rjb_field = widgets.Text(
        value='',
        placeholder='e.g., 40 | maximum distance of record is 40 km',
        description='Rrup:',
        disabled=False,
        style=style,
        layout = layout

    )
    ScaleFactor_field = widgets.Text(
        value='',
        placeholder='e.g., 0.25-4 | range is 0.25 to 4',
        description='Range of scaling factor:',
        disabled=False,
        style=style,
        layout = layout

    )
    maxRecEvt_field = widgets.Text(
        value='',
        placeholder='e.g., 2 | maximum number of records for one event',
        description='Maximum number of records that can be selected for one event:',
        disabled=False,
        style=style,
        layout = layout

    )
    numPulse_field = widgets.Text(
        value='',
        placeholder='e.g., 7 | 7 pulse records are to be selected (leave blank if unconstrained)',
        description='Number of pulse records in a suite:',
        disabled=False,
        style=style,
        layout = layout

    )
    minPulseT_field = widgets.Text(
        value='',
        placeholder='e.g., 3 | minimum pulse period of the selected pulse records is 3s (leave blank if unconstrained)',
        description='Minimum pulse period required for selected pulse records:',
        disabled=False,
        style=style,
        layout = layout

    )
    minDuration_field = widgets.Text(
        value='',
        placeholder='e.g., 10 | minimum duration of the selected non-pulse records is 10s (leave blank if unconstrained)',
        description='Minimum 5-95% Arias Intensity duration required for selected non-pulse records:',
        disabled=False,
        style=style,
        layout = layout

    )
    ductility_checkList = widgets.SelectMultiple(
        options=['Ductility = 1 (TargetSpectra_Elastic.csv)',
          'Ductility = 1.5 (TargetSpectra_ductility1.5.csv)',
          'Ductility = 2 (TargetSpectra_ductility2.0.csv)',
          'Ductility = 3 (TargetSpectra_ductility3.0.csv)',
          'Ductility = 4 (TargetSpectra_ductility4.0.csv)',
          'Ductility = 5 (TargetSpectra_ductility5.0.csv)'],
        value=['Ductility = 1 (TargetSpectra_Elastic.csv)'],
        description='Ductility level of target(s) \n hold Ctrl to have multiple options:',
        disabled=False,
        style=style,
        layout = layout,
    )
    ductility_checkList.layout.visibility = 'hidden'
    ductility_checkList.layout.display = 'none'
    button_run_gm_selection = widgets.Button(description = 'Step 2: Run Ground Motion Selection Function', layout=layout_button1)
    button_plot_selection = widgets.Button(description = 'Step 3: Generate Report of Selected Motions', layout=layout_button2)

    return [fileLoc_field,Elastic_field,numGM_field,magnitude_field,rjb_field,ScaleFactor_field,maxRecEvt_field,
            numPulse_field,minPulseT_field,minDuration_field,ductility_checkList,
            button_run_gm_selection,button_plot_selection]
'''


def setUpInputInterface():
    # custom_css = """
    # <style>
    #     .jupyter-widgets button {
    #         font-size: 20px;
    #     }
    #     # .widget-dropdown select  {
    #     #     font-size: 16px; /* Adjust font size as needed */
    #     # }
    #     /* Targeting dropdown options */
    #     .widget-dropdown select, .widget-select-multiple select {
    #         font-size: 16px; 
    #     }        
    #     # /* Targeting options within the dropdown */
    #     # .widget-dropdown .widget-select > select > option, .widget-select-multiple select > option {
    #     #     font-size: 16px; /* This ensures the options have the desired font size */
    #     # }
    # </style>
    # """
    custom_css = """
    <style>
        .jupyter-widgets button {
            font-size: 20px;
        }
        /* Targeting dropdown options */
        .widget-dropdown select, .widget-select-multiple select {
            font-size: 16px; 
        }        
    </style>
    """
    display(HTML(custom_css))

    style = {'description_width': 'initial', 'font_size': '16px'}
    layout = {'width': '100%', 'max_width': '1500px'}
    # label_style = '<style>.label {font-size: 16px; margin-right: 5px;}</style>'
    # label_layout = Layout(width='400px')
    layout_button1 = {'width': 'auto', 'max_width': '1500px', 'margin': '25px 0 0 0', 'height': '35px'}
    layout_button2 = {'width': 'auto', 'max_width': '1500px', 'margin': '8px 0 0 0', 'height': '35px'}
    layout_button3 = {'width': 'auto', 'max_width': '1500px', 'margin': '8px 0 0 0', 'height': '35px'}

    max_length = 70  # Max length of description to align all widgets

    def toggle_widgets(change):
        """
        Toggle the visibility of widgets based on the value of Elastic_field.
        """
        if change['new'] == 'Elastic (TargetSpectra_Elastic.csv)':
            numPulse_field_widget.layout.visibility = 'visible'
            minPulseT_field_widget.layout.visibility = 'visible'
            minDuration_field_widget.layout.visibility = 'visible'
            ductility_checkList_widget.layout.visibility = 'hidden'
            numPulse_field_widget.layout.display = 'flex'
            minPulseT_field_widget.layout.display = 'flex'
            minDuration_field_widget.layout.display = 'flex'
            ductility_checkList_widget.layout.display = 'none'
        else:
            numPulse_field_widget.layout.visibility = 'hidden'
            minPulseT_field_widget.layout.visibility = 'hidden'
            minDuration_field_widget.layout.visibility = 'hidden'
            ductility_checkList_widget.layout.visibility = 'visible'
            numPulse_field_widget.layout.display = 'none'
            minPulseT_field_widget.layout.display = 'none'
            minDuration_field_widget.layout.display = 'none'
            ductility_checkList_widget.layout.display = 'flex'

    def create_widget_with_custom_label(description, widget, font_size='16px'):
        custom_label = widgets.HTML(
            value=f'<div style="font-size: {font_size}; font-weight: bold; margin-right: 10px;">{description}</div>')
        hbox = widgets.HBox([custom_label, widget], layout=Layout(display='flex', width='100%', max_width='1500px'))
        custom_label.layout.flex = '0 1 auto'  # Label can grow and shrink but starts with content size
        widget.layout.flex = '1'  # Widget takes up the remaining space
        return hbox

    # Define your widgets without the description attribute
    fileLoc_field = widgets.Text(value=f'{os.getcwd()}/Inputs', placeholder=f'{os.getcwd()}/Inputs', disabled=True,
                                 style=style, layout=layout)
    Elastic_field = widgets.Dropdown(options=['Elastic (TargetSpectra_Elastic.csv)', 'Inelastic'],
                                     value='Elastic (TargetSpectra_Elastic.csv)', style=style, layout=layout)
    numGM_field = widgets.Text(value='', placeholder='e.g., 11 | 11 ground motions to be selected in a suite',
                               style=style, layout=layout)
    magnitude_field = widgets.Text(value='', placeholder='e.g., 6-8 | magnitude range is 6 to 8', style=style,
                                   layout=layout)
    rjb_field = widgets.Text(value='', placeholder='e.g., 40 | maximum distance of record is 40 km', style=style,
                             layout=layout)
    ScaleFactor_field = widgets.Text(value='', placeholder='e.g., 0.25-4 | scaling factor range is 0.25 to 4',
                                     style=style, layout=layout)
    maxRecEvt_field = widgets.Text(value='', placeholder='e.g., 2 | maximum number of records for one event',
                                   style=style, layout=layout)
    numPulse_field = widgets.Text(value='', placeholder='e.g., 7 | 7 pulse records are to be selected (leave blank if unconstrained)', style=style, layout=layout)
    minPulseT_field = widgets.Text(value='', placeholder='e.g., 3 | minimum pulse period of the selected pulse records is 3s (leave blank if unconstrained)', style=style, layout=layout)
    minDuration_field = widgets.Text(value='', placeholder='e.g., 10 | minimum duration of the selected non-pulse records is 10s (leave blank if unconstrained)', style=style, layout=layout)
    ductility_checkList = widgets.SelectMultiple(options=['Ductility = 1 (TargetSpectra_Elastic.csv)',
                                                          'Ductility = 1.5 (TargetSpectra_ductility1.5.csv)',
                                                          'Ductility = 2 (TargetSpectra_ductility2.0.csv)',
                                                          'Ductility = 3 (TargetSpectra_ductility3.0.csv)',
                                                          'Ductility = 4 (TargetSpectra_ductility4.0.csv)',
                                                          'Ductility = 5 (TargetSpectra_ductility5.0.csv)'],
                                                 value=['Ductility = 1 (TargetSpectra_Elastic.csv)'], style=style,
                                                 layout=layout)

    button_run_gm_selection = widgets.Button(description='Step 2: Run Ground Motion Selection Function',
                                             layout=layout_button1)
    button_plot_selection = widgets.Button(description='Step 3: Generate Report of Selected Motions',
                                           layout=layout_button2)
    ##Yunbo
    user_name = widgets.Text(value='', placeholder='Your registered email for NGA-West2', style=style,
                             layout=layout)
    password = widgets.Password(value='', placeholder='Your password', style=style,
                                layout=layout)
    user_name_widget = create_widget_with_custom_label("Email:", user_name)
    password_widget = create_widget_with_custom_label("Password:", password)

    button_data_download = widgets.Button(description='Step 4 (Only for Chrome): Download the data from the NGA-West2 website',
                                          layout=layout_button3)

    # Use the custom label function for all widgets
    fileLoc_field_widget = create_widget_with_custom_label("File path to elastic target folder:", fileLoc_field)
    Elastic_field_widget = create_widget_with_custom_label("Consider only elastic target or inelastic target:",
                                                           Elastic_field)
    numGM_field_widget = create_widget_with_custom_label("Number of records in a suite:", numGM_field)
    magnitude_field_widget = create_widget_with_custom_label("M:", magnitude_field)
    rjb_field_widget = create_widget_with_custom_label("Rrup:", rjb_field)
    ScaleFactor_field_widget = create_widget_with_custom_label("Range of scaling factor:", ScaleFactor_field)
    maxRecEvt_field_widget = create_widget_with_custom_label(
        "Maximum number of records that can be selected for one event:", maxRecEvt_field)
    numPulse_field_widget = create_widget_with_custom_label("Number of pulse records in a suite:", numPulse_field)
    minPulseT_field_widget = create_widget_with_custom_label(
        "Minimum pulse period required for selected pulse records:", minPulseT_field)
    minDuration_field_widget = create_widget_with_custom_label(
        "Minimum 5-95% Arias Intensity duration required for selected non-pulse records:", minDuration_field)
    ductility_checkList_widget = create_widget_with_custom_label(
        "Ductility level of target(s) \n hold Ctrl to have multiple options:", ductility_checkList)

    Elastic_field.observe(toggle_widgets, names='value')
    ductility_checkList_widget.layout.visibility = 'hidden'
    ductility_checkList_widget.layout.display = 'none'

    return [fileLoc_field_widget, Elastic_field_widget, numGM_field_widget, magnitude_field_widget, rjb_field_widget,
            ScaleFactor_field_widget, maxRecEvt_field_widget,
            numPulse_field_widget, minPulseT_field_widget, minDuration_field_widget, ductility_checkList_widget,
            button_run_gm_selection, button_plot_selection, button_data_download, user_name_widget, password_widget]


#######################################################
# [fileLoc_field,Elastic_field,numGM_field,magnitude_field,rjb_field,ScaleFactor_field,maxRecEvt_field,
#             numPulse_field,minPulseT_field,minDuration_field,ductility_checkList,button_run_gm_selection,
#             button_plot_selection,button_data_download] = setUpInputInterface()

###########################################################


def clearOutputs():
    outputspath = f'{os.getcwd()}/Outputs/'
    try:
        for filename in ['cor_selected motions.png', 'Selected motions.png', 'selected_motions.csv']:
            os.remove(os.path.join(outputspath, filename))
    except:
        pass


def download_file_backend(email_s, password):
    # chrome_options = Options()
    #
    # prefs = {"download.default_directory": link}
    # chrome_options.add_experimental_option("prefs", prefs)
    # if brower_jump == False:
    #     chrome_options.add_argument("--headless")

    ####################################################################
    # html_widget = widgets.HTML(
    #     value='<div class="message-container" style="font-size: 20px;">Username and Password have been recorded</div>'
    # )

    # message_label = widgets.HTML(
    #     # value='<div class="message-container">Username and Password have been recorded</div>'
    #     value='<div class="message-container" style="font-size: 20px;">Username and Password have been recorded</div>'
    # )

    # container = widgets.HBox([message_label], layout=widgets.Layout(align_items='center'))
    # display(container)
    ######################################################################
    # if headless == True:
    #     options = Options()
    #     options.add_argument("--headless")
    #     options.add_argument("--no-sandbox")  # Bypass OS security model
    #     options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    #     options.add_argument("--disable-gpu")
    #     options.add_argument('--remote-debugging-pipe')
    #     browser = webdriver.Chrome(options=options)
    # else:
    #     browser = webdriver.Chrome()
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")  # Bypass OS security model
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    options.add_argument("--disable-gpu")
    # options.add_argument('--remote-debugging-pipe')
    browser = webdriver.Chrome(options=options)

    df = pd.read_csv(os.getcwd() + '/Outputs/selected_motions.csv')
    RSN_numbers = (df.loc[:, 'RSN']).values
    RSN_number = ''
    for i in RSN_numbers:
        RSN_number = RSN_number + str(i) + ','

    message_label1 = widgets.HTML(
        # value='<div class="message-container">Username and Password have been recorded</div>'
        value='<div class="message-container" style="font-size: 20px;">RSN number has been get.</div>'
    )

    container1 = widgets.HBox([message_label1], layout=widgets.Layout(align_items='center'))
    display(container1)

    browser.get('https://ngawest2.berkeley.edu/spectras/new?sourceDb_flag=1')
    email = browser.find_element(By.ID, "user_email")
    pw = browser.find_element(By.ID, "user_password")
    email.send_keys(email_s)
    # email.send_keys(Keys.RETURN)
    pw.send_keys(password)
    # pw.send_keys(Keys.RETURN)
    sign_in = browser.find_element(By.XPATH, '// *[ @ id = "user_submit"]')
    sign_in.click()
    time.sleep(2)
    ##### chech password
    alert = browser.find_element(By.XPATH, '//*[@id="content"]/p')

    if alert.text == 'Invalid email or password.':
        message_label2 = widgets.HTML(
            # value='<div class="message-container">Username and Password have been recorded</div>'
            value='<div class="message-container" style="font-size: 20px;">Unable to sign in: Invalid email or password. Please confirm them.</div>'
        )
        container2 = widgets.HBox([message_label2], layout=widgets.Layout(align_items='center'))
        return (display(container2))

    if_success = browser.find_element(By.XPATH, '// *[ @ id = "notice"]')


    if if_success.text == 'Signed in successfully.':
        message_label2 = widgets.HTML(
            # value='<div class="message-container">Username and Password have been recorded</div>'
            value='<div class="message-container" style="font-size: 20px;">Signed in to the NGA-West2 Successfully.</div>'
        )
        container2 = widgets.HBox([message_label2], layout=widgets.Layout(align_items='center'))
        display(container2)

    ####
    # wait for the submit button
    submit = browser.find_element(By.XPATH, "//div[@id = 'buttons']/button")
    submit.click()
    time.sleep(5)

    # wait for search
    RSN = browser.find_element(By.ID, "search_search_nga_number")
    RSN.send_keys(RSN_number)
    search = browser.find_element(By.XPATH, "//div[@class = 'left_column']/fieldset/button")
    search.click()
    time.sleep(5)
    # print("RSN Number is input")
    #
    # #### download
    # print("Download is initializing")

    message_label3 = widgets.HTML(
        # value='<div class="message-container">Username and Password have been recorded</div>'
        value='<div class="message-container" style="font-size: 20px;">RSN Number is input and Download is initializing.</div>'
    )

    container3 = widgets.HBox([message_label3], layout=widgets.Layout(align_items='center'))
    display(container3)
    ##
    download = browser.find_element(By.XPATH, '//*[@id="middle_submit"]/fieldset[3]/button[2]')
    time.sleep(2)
    download.click()
    ## alert from the NGA-WEST2
    alert1 = browser.switch_to.alert
    # print(alert1.text)
    message_label_Alert1 = widgets.HTML(
        value=f'<div class="message-container" style="font-size: 18px;">{alert1.text}</div>'
    )

    container_Alert1 = widgets.HBox([message_label_Alert1], layout=widgets.Layout(align_items='center'))
    display(container_Alert1)
    alert1.accept()

    alert2 = browser.switch_to.alert
    # print(alert2.text)
    message_label_Alert2 = widgets.HTML(
        value=f'<div class="message-container" style="font-size: 18px;">{alert2.text}</div>'
    )
    container_Alert2 = widgets.HBox([message_label_Alert2], layout=widgets.Layout(align_items='center'))
    display(container_Alert2)
    alert2.accept()
    # print("Please wait until the download is finished")
    message_label4 = widgets.HTML(
        # value='<div class="message-container">Username and Password have been recorded</div>'
        value='<div class="message-container" style="font-size: 20px;">Please wait until the download is finished...</div>'
    )

    container4 = widgets.HBox([message_label4], layout=widgets.Layout(align_items='center'))
    display(container4)

    time.sleep(6)

    message_label5 = widgets.HTML(
        # value='<div class="message-container">Username and Password have been recorded</div>'
        value='<div class="message-container" style="font-size: 20px;">The download has been finished. The downloaded file name is "PEERNGARecords_Unscaled.zip".</div>'
    )

    container5 = widgets.HBox([message_label5], layout=widgets.Layout(align_items='center'))
    display(container5)

    return


def get_username_password(user_name_widget, password_widget):
    if user_name_widget.children[1].value:
        username = str(user_name_widget.children[1].value)
    else:
        message_label = widgets.HTML(
            value='<div class="message-container">please type in your registered email on NGA-West2</div>')
        container = widgets.HBox([message_label], layout=widgets.Layout(align_items='center'))
        return display(container)

    if password_widget.children[1].value:
        password = str(password_widget.children[1].value)
    else:
        message_label = widgets.HTML(value='<div class="message-container">please type in your password</div>')
        container = widgets.HBox([message_label], layout=widgets.Layout(align_items='center'))
        return display(container)

    message_label = widgets.HTML(
        # value='<div class="message-container">Username and Password have been recorded</div>'
        value='<div class="message-container" style="font-size: 20px;">Username and Password have been recorded</div>'
    )
    # html_widget = widgets.HTML(
    #     value='<div class="message-container" style="font-size: 20px;">Username and Password have been recorded</div>'
    # )

    # html_widget = widgets.HTML(
    #     value='<div class="message-container" style="font-size: 20px; color: red; font-family: Arial;">Username and Password have been recorded</div>'
    # )

    container = widgets.HBox([message_label], layout=widgets.Layout(align_items='center'))
    display(container)

    # download_file(username, password, headless=True)

    return username, password


def generateInput(Elastic_field_widget, numGM_field_widget, magnitude_field_widget, rjb_field_widget,
                  ScaleFactor_field_widget, maxRecEvt_field_widget, numPulse_field_widget, minPulseT_field_widget,
                  minDuration_field_widget, ductility_checkList_widget):
    Searchparameters = {
        "type": None,
        "numGM": None,
        "minMag": None,
        "maxMag": None,
        "maxRjb": None,
        "minSF": None,
        "maxSF": None,
        "maxRecEvt": None,
        "numPulse": None,
        "minPulseT": None,
        "minDuration": None
    }
    if Elastic_field_widget.children[1].value:
        Searchparameters['type'] = Elastic_field_widget.children[1].value
    else:
        # Default elastic
        Searchparameters['type'] = 'Elastic (TargetSpectra_Elastic.csv)'
    if numGM_field_widget.children[1].value:
        A = numGM_field_widget.children[1].value
        Searchparameters['numGM'] = float(A)
    else:
        # Default value: 11
        Searchparameters['numGM'] = 11
    if magnitude_field_widget.children[1].value:
        try:
            A = magnitude_field_widget.children[1].value.split("-")
            Searchparameters['minMag'] = float(A[0])
            Searchparameters['maxMag'] = float(A[1])
        except:
            return "Please go back to Step 1 and check input magnitude range."
    else:
        # Default value: 6.0-8
        Searchparameters['minMag'] = 6.0
        Searchparameters['maxMag'] = 8.0
    if rjb_field_widget.children[1].value:
        A = rjb_field_widget.children[1].value
        Searchparameters['maxRjb'] = float(A)
    else:
        # Default value: 0-40km
        Searchparameters['maxRjb'] = 40
    if ScaleFactor_field_widget.children[1].value:
        try:
            A = ScaleFactor_field_widget.children[1].value.split("-")
            Searchparameters['minSF'] = float(A[0])
            Searchparameters['maxSF'] = float(A[1])
        except:
            return "Please go back to Step 1 and check input scaling factor range."
    else:
        # Default value: 0.25-4.0
        Searchparameters['minSF'] = 0.25
        Searchparameters['maxSF'] = 4
    if maxRecEvt_field_widget.children[1].value:
        A = maxRecEvt_field_widget.children[1].value
        Searchparameters['maxRecEvt'] = float(A)
    else:
        # Default value: 2
        Searchparameters['maxRecEvt'] = 2

    if Elastic_field_widget.children[1].value == 'Elastic (TargetSpectra_Elastic.csv)':
        if numPulse_field_widget.children[1].value:
            A = numPulse_field_widget.children[1].value
            Searchparameters['numPulse'] = float(A)
        else:
            # Default value: None
            Searchparameters['numPulse'] = None
        if minPulseT_field_widget.children[1].value:
            A = minPulseT_field_widget.children[1].value
            Searchparameters['minPulseT'] = float(A)
        else:
            # Default value: None
            Searchparameters['minPulseT'] = None
        if minDuration_field_widget.children[1].value:
            A = minDuration_field_widget.children[1].value
            Searchparameters['minDuration'] = float(A)
        else:
            # Default value: None
            Searchparameters['minDuration'] = None
        Searchparameters['targetSpectraFileNames'] = ['TargetSpectra_Elastic']
        Searchparameters['databaseFileNames'] = ['Horizontal_elastic_PSA']
    else:
        A = ductility_checkList_widget.children[1].value
        databaseFileNames_all = ['Horizontal_elastic_PSA', 'Horizontal_inelastic_1.5', 'Horizontal_inelastic_2',
                                 'Horizontal_inelastic_3', 'Horizontal_inelastic_4', 'Horizontal_inelastic_5']
        targetSpectraFileNames_all = ['TargetSpectra_Elastic', 'TargetSpectra_ductility1.5', 'TargetSpectra_ductility2.0',
                                      'TargetSpectra_ductility3.0', 'TargetSpectra_ductility4.0',
                                      'TargetSpectra_ductility5.0']
        targetSpectraFileNames = []
        databaseFileNames = []
        for a in A:
            target_file = a[a.find("(") + 1:a.find(")") - 4]
            targetSpectraFileNames.append(target_file)
            database_file = databaseFileNames_all[targetSpectraFileNames_all.index(target_file)]
            databaseFileNames.append(database_file)
        Searchparameters['targetSpectraFileNames'] = targetSpectraFileNames
        Searchparameters['databaseFileNames'] = databaseFileNames
    try:
        with open("./Inputs/SearchParameters.json", "w") as file:
            json.dump(Searchparameters, file, indent=4)
        return ""  # "Success: Search parameters saved!"
    except Exception as e:
        return f"Error: {str(e)}"


def read_search_parameters():
    with open("./Inputs/SearchParameters.json", "r") as file:
        parameters = json.load(file)

    warning_list = []
    if parameters['numGM'] >= 50:
        warning_message = 'Too many number of records in a suite, please check it.'
        warning_list.append(warning_message)
    if parameters['minMag'] >= 7.9:
        warning_message = 'Input magnitude out of range, please check it.'
        warning_list.append(warning_message)

    return parameters, warning_list


def display_spinner_and_label():
    spinner_html = widgets.HTML("""
    <style>
    .message-container {
        display: flex;
        flex-direction: column;
        font-size: 20px; /* Enlarge the message text */
    }
    .spinner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 50px; /* Adjust as needed */
        height: 50px; /* Adjust to match the message height if needed */
    }
    @keyframes spinner {
        to {transform: rotate(360deg);}
    }
    .spinner {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 3px solid rgba(195, 195, 195, 0.6);
        border-radius: 50%;
        border-top-color: #3498db; /* Blue */
        animation: spinner 1s ease-in-out infinite;
        -webkit-animation: spinner 1s ease-in-out infinite;
    }
    </style>
    <div class="spinner-container">
        <div class="spinner"></div>
    </div>
    """)
    message_label = widgets.HTML(value='<div class="message-container">Running ground motion selection...</div>')

    # Use HBox for horizontal layout but with alignment adjustments
    container = widgets.HBox([message_label, spinner_html], layout=widgets.Layout(align_items='center'))

    display(container)
    return spinner_html, message_label


def run_gm_selection(search_params, warning_list):
    def run_task():
        try:
            GMS_analysis.run_ground_motion_selection()
            message_label.value = '<div class="message-container">Ground motion selection finished, please run Step 3.</div>'
        except Exception as e:
            message_label.value = f'<div class="message-container">Error during ground motion selection: {str(e)}</div>'
        finally:
            spinner_html.layout.display = 'none'

    onlyElastic = (search_params['type'] == 'Elastic (TargetSpectra_Elastic.csv)')
    GMS_analysis = search_ground_motion_CMS()
    GMS_analysis.nGM = int(search_params['numGM'])
    GMS_analysis.MScenario = 0.5 * (search_params['minMag'] + search_params['maxMag'])
    GMS_analysis.Msd = 0.5 * (search_params['maxMag'] - search_params['minMag'])
    GMS_analysis.RMax = search_params['maxRjb']
    GMS_analysis.minScale = search_params['minSF']
    GMS_analysis.maxScale = search_params['maxSF']
    GMS_analysis.MaxNoGMsFromOneEvent = search_params['maxRecEvt']
    GMS_analysis.targetSpectraFileNames = search_params['targetSpectraFileNames']
    GMS_analysis.databaseFileNames = search_params['databaseFileNames']
    if onlyElastic:
        GMS_analysis.only_elastic = onlyElastic
        GMS_analysis.nPulse = int(search_params['numPulse']) if search_params['numPulse'] is not None else None
        GMS_analysis.TPulsemin = search_params['minPulseT']
        GMS_analysis.DuraMin = search_params['minDuration']

    spinner_html, message_label = display_spinner_and_label()
    if len(warning_list) > 0:
        warnings_html = "<ul>" + "".join(f"<li>{warning}</li>" for warning in warning_list) + "</ul>"
        warnings_text = "Ground motion selection not run due to warnings:" + warnings_html
        additional_instruction = "Please correct the above issues and go back to Step 1."
        message_label.value = f'<div class="message-container">{warnings_text}{additional_instruction}</div>'
        spinner_html.layout.display = 'none'
    else:
        thread = threading.Thread(target=run_task)
        thread.start()


def summarize_data(df):
    cols_to_select = ['RSN', 'EQID', 'Magnitude', 'Rjb', 'Vs30', 'scale', 'Duration_5_95', 'Tp', 'EQname',
                      'StationName']
    cols_new_names = ['RSN', 'EQID', 'Magnitude', 'Rrup', 'Vs30', 'SF', 'Duration', 'Tpulse (s)', 'EQname',
                      'StationName']

    dfsum = df[cols_to_select]
    dfsum.columns = cols_new_names
    dfsum = dfsum.assign(
        Vs30=dfsum['Vs30'].astype(int),
        SF=dfsum['SF'].round(2),
        EQID=dfsum['EQID'].astype(int),
        Duration=dfsum['Duration'].round(2),
        **{'Tpulse (s)': dfsum['Tpulse (s)'].round(2)}
    )

    return dfsum


def plot_selected_records_table(dfsum):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    header_colors = ['lightblue'] * len(dfsum.columns)
    table_data = [dfsum.columns.tolist()] + dfsum.values.tolist()
    table_data[1:] = [[int(sublist[0]), int(sublist[1])] + sublist[2:4] + [int(sublist[4])] + sublist[5:] for sublist in
                      table_data[1:]]
    cell_colors = [header_colors] + [['white'] * len(dfsum.columns)] * len(dfsum)
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', \
                     cellColours=cell_colors)

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    cell_dict = table.get_celld()
    for key in cell_dict:
        cell = cell_dict[key]
        cell.set_height(0.05)  # Change row height as desired

    col_widths = [0.1, 0.1, 0.15, 0.12, 0.1, 0.1, 0.14, 0.15, 0.4, 0.5]  # Adjust specific column widths
    for i, width in enumerate(col_widths):
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)
    plt.show()


def plot_selected_records(df, search_params):
    for target_file, database_file in zip(search_params['targetSpectraFileNames'], search_params['databaseFileNames']):
        target = pd.read_csv('./Inputs/' + target_file + '.csv')
        PSA_Table = pd.read_csv('./Datasets/' + database_file + '.csv')
        target_periods = target['T']
        target_mean = target['Mean']
        target_sd = target['Sd']
        RSN_list = df['RSN']
        SF_list = df['scale']
        period_list = [period for period in PSA_Table.columns[1:] if float(period) <= max(target_periods)]
        period_float = np.array(period_list).astype(float)

        fig = plt.figure(figsize=(14, 5))
        fig.suptitle(f"Compare Selected Ground Motion Suite with Target: {target_file.split('_')[1]}", fontsize=14)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224, sharex=ax2)

        psa_array = []
        for RSN, SF in zip(RSN_list, SF_list):
            record_i = PSA_Table.loc[PSA_Table['RSN'] == RSN, period_list].values[0]
            if RSN == RSN_list.iloc[0]:
                label = 'Scaled selected records'
            else:
                label = ''
            ax1.loglog(period_float, record_i * SF, c='grey', label=label)
            psa_array.append(record_i * SF)

        psa_array = np.array(psa_array)
        log_psa = np.log(psa_array)
        suitemean = np.exp(np.mean(log_psa, axis=0))
        suitestd = np.std(log_psa, axis=0)

        ax1.loglog(period_float, suitemean, c='r', label='Selected records mean', linewidth=2.5)
        ax1.loglog(period_float, np.exp(np.log(suitemean) - suitestd), c='r', linestyle='--',
                   label='Selected records mean+/-1SD', linewidth=2.5)
        ax1.loglog(period_float, np.exp(np.log(suitemean) + suitestd), c='r', linestyle='--', linewidth=2.5)
        ax3.semilogx(period_float, suitestd, label='Selected records SD')

        ax1.loglog(target_periods, target_mean, linestyle='-', c='k', label='Target mean', linewidth=2.5)
        ax1.loglog(target_periods, np.exp(np.log(target_mean) - target_sd), linestyle='--', c='k',
                   label='Target mean+/-1SD', linewidth=2.5)
        ax1.loglog(target_periods, np.exp(np.log(target_mean) + target_sd), linestyle='--', c='k', linewidth=2.5)
        ax3.semilogx(target_periods, target_sd, label='Target sd')

        targetmean_interp = np.exp(np.interp(np.log(period_float), np.log(target_periods), np.log(target_mean)))
        ax2.semilogx(period_float[period_float >= min(target_periods)],
                     suitemean[period_float >= min(target_periods)] / targetmean_interp[
                         period_float >= min(target_periods)], c='r', label='Mean')

        for ax in [ax1, ax2, ax3]:
            ax.legend()
            ax.grid()
            ax.set_xlabel('Period (s)')
        ax1.set_ylabel('RotD50 PSA (g)')
        ax2.set_ylabel('Suite/target Ratio')
        ax3.set_ylabel('StDev')
        ax2.set_ylim([0.75, 1.25])
        ax3.set_ylim([0, 1.0])

    plt.show()


def plot_selection(output_file_path, search_params):
    df = pd.read_csv(output_file_path)
    dfsum = summarize_data(df)
    plot_selected_records_table(dfsum)
    plot_selected_records(df, search_params)
