import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import json
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from main import camToVcsHO_HP, camToVcsGaze  # Import necessary functions
import MPIC_Data_Transformations_V6_7 as MPIC_Data_Transformations
 
#---------------------------------------------------function defination start-----------------------------------------------------------------

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
        return None
    

def plot_rectangles_3d(df):
    """
    Plots rectangles in 3D space by connecting points A, B, C, and D for each row in the CSV file.

    Parameters:
    csv_file_path (str): Path to the CSV file containing the coordinates of points A, B, C, and D.
    """
    # Step 1: Read the CSV file into a DataFrame
   

    # Step 2: Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(df)):
        # Coordinates of points A, B, C, D
        A = [df['Ax'][i], df['Ay'][i], df['Az'][i]]
        B = [df['Bx'][i], df['By'][i], df['Bz'][i]]
        C = [df['Cx'][i], df['Cy'][i], df['Cz'][i]]
        D = [df['Dx'][i], df['Dy'][i], df['Dz'][i]]
        
        # Plot points
        ax.scatter(*A, color='r', marker='o')
        ax.scatter(*B, color='g', marker='o')
        ax.scatter(*C, color='b', marker='o')
        ax.scatter(*D, color='y', marker='o')
        
        # Connect points with lines to form a rectangle
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='black')
        ax.plot([B[0], C[0]], [B[1], C[1]], [B[2], C[2]], color='black')
        ax.plot([C[0], D[0]], [C[1], D[1]], [C[2], D[2]], color='black')
        ax.plot([D[0], A[0]], [D[1], A[1]], [D[2], A[2]], color='black')

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Display the plot
    plt.show()



def round_comma_number(value):
    try:
        # Replace comma with a dot for Python to understand the number correctly
        value = str(value).replace(',', '.')
        # Convert to float and round the value
        value = round(float(value))
        # Replace the dot back with a comma
        return str(value).replace('.', ',')
    except ValueError:
        return value

def round_numerical_columns(df):
    return df.applymap(round_comma_number)

def calculate_points(df):

    y = np.radians(df['Yaw_Pred'])
    p = np.radians(df['Pitch_Pred'])
    L = -2000  # Assuming a fixed value for L

    T1 = np.cos(y) * np.cos(p) * L
    T2 = np.sin(y) * np.cos(p) * L
    T3 = -np.sin(p) * L

    numerator = df['Intr_Plane_d'] - (
        (df['Intr_Plane_n1'] * df['HP_X_Pred']) +
        (df['Intr_Plane_n2'] * df['HP_Y_Pred']) +
        (df['Intr_Plane_n3'] * df['HP_Z_Pred'])
    )
    
    denominator = (
        (df['Intr_Plane_n1'] * T1) +
        (df['Intr_Plane_n2'] * T2) +
        (df['Intr_Plane_n3'] * T3)
    )

    F1 = numerator / denominator

    x = df['HP_X_Pred'] + (T1 * F1)
    y = df['HP_Y_Pred'] + (T2 * F1)
    z = df['HP_Z_Pred'] + (T3 * F1)

    df['Point_x'] = x.fillna(0).round(0).astype(int)
    df['Point_y'] = y.fillna(0).round(0).astype(int)
    df['Point_z'] = z.fillna(0).round(0).astype(int)
    

    return df    # Existing calculate_points function...
    # Your existing code here
    pass

def merge_data(file1, file2):
    file1['First_Char'], file1['ROI_Num'] = zip(*file1['Fragment'].apply(extract_fragment_details))
    file2['ROI_Num'] = file2['ROI_x'].str.split('_').str[0]

   
    # Merge the dataframes based on the first character and ROI number
    merged_df = pd.merge(file1, file2, on='ROI_Num', how='left')
   
    # Drop the temporary columns
    merged_df.drop(columns=['First_Char', 'ROI_Num'], inplace=True)
   
    return merged_df

def create_roi_dict_lhd(df):
    roi_dict_lhd = {}

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        roi = str(row['Processed_Fragment_ROI'])  # Ensure it's a string
        fragment = row['Processed_Fragment']
        roi_value = row['LHD']
        
        # If this Processed_Fragment_ROI is not in the dictionary, add it
        if roi not in roi_dict_lhd:
            roi_dict_lhd[roi] = {}
        
        # Add the Processed_Fragment and its corresponding ROI_x value to the dictionary
        roi_dict_lhd[roi][fragment] = roi_value
    
    return roi_dict_lhd

def create_roi_dict_rhd(df):
    roi_dict_rhd = {}

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        roi = str(row['Processed_Fragment_ROI'])  # Ensure it's a string
        fragment = row['Processed_Fragment']
        roi_value = row['RHD']
        
        # If this Processed_Fragment_ROI is not in the dictionary, add it
        if roi not in roi_dict_rhd:
            roi_dict_rhd[roi] = {}
        
        # Add the Processed_Fragment and its corresponding ROI_x value to the dictionary
        roi_dict_rhd[roi][fragment] = roi_value
    
    return roi_dict_rhd


def calculate_point_d(df):
    # Calculate Dx, Dy, Dz for each row
    df['Dx'] = df['Ax'] + (df['Cx'] - df['Bx'])
    df['Dy'] = df['Ay'] + (df['Cy'] - df['By'])
    df['Dz'] = df['Az'] + (df['Cz'] - df['Bz'])
    # df['Dx_p'] = df['Ax_p'] + (df['Cx_p'] - df['Bx_p'])
    # df['Dy_p'] = df['Ay_p'] + (df['Cy_p'] - df['By_p'])
    # df['Dz_p'] = df['Az_p'] + (df['Cz_p'] - df['Bz_p'])
    return df

def calculate_point_dx_p(df):
    # Convert columns to floats
    df['Ax_p'] = pd.to_numeric(df['Ax_p'], errors='coerce')
    df['Ay_p'] = pd.to_numeric(df['Ay_p'], errors='coerce')
    df['Az_p'] = pd.to_numeric(df['Az_p'], errors='coerce')
    df['Bx_p'] = pd.to_numeric(df['Bx_p'], errors='coerce')
    df['By_p'] = pd.to_numeric(df['By_p'], errors='coerce')
    df['Bz_p'] = pd.to_numeric(df['Bz_p'], errors='coerce')
    df['Cx_p'] = pd.to_numeric(df['Cx_p'], errors='coerce')
    df['Cy_p'] = pd.to_numeric(df['Cy_p'], errors='coerce')
    df['Cz_p'] = pd.to_numeric(df['Cz_p'], errors='coerce')

    # Calculate Dx, Dy, Dz for each row
    df['Dx_p'] = df['Ax_p'] + (df['Cx_p'] - df['Bx_p'])
    df['Dy_p'] = df['Ay_p'] + (df['Cy_p'] - df['By_p'])
    df['Dz_p'] = df['Az_p'] + (df['Cz_p'] - df['Bz_p'])
    
    return df



def process_csv_roi(input_df):
    output_data = []

    unique_rois = input_df['Processed_Fragment_ROI'].unique()

    for roi in unique_rois:
        Ax = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_A'), drive_style].values[0]
        Ay = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_A'), drive_style].values[0]
        Az = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_A'), drive_style].values[0]
        Bx = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_B'), drive_style].values[0]
        By = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_B'), drive_style].values[0]
        Bz = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_B'), drive_style].values[0]
        Cx = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_C'), drive_style].values[0]
        Cy = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_C'), drive_style].values[0]
        Cz = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_C'), drive_style].values[0]

        output_data.append({
            'ROI': roi,
            'Ax': Ax,
            'Ay': Ay,
            'Az': Az,
            'Bx': Bx,
            'By': By,
            'Bz': Bz,
            'Cx': Cx,
            'Cy': Cy,
            'Cz': Cz
        })

    return pd.DataFrame(output_data)


def process_csv(input_df):
    output_data = []

    unique_rois = input_df['Processed_Fragment_ROI'].unique()

    for roi in unique_rois:
        Ax = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_A'), 'cad_rhd'].values[0]
        Ay = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_A'), 'cad_rhd'].values[0]
        Az = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_A'), 'cad_rhd'].values[0]
        Bx = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_B'), 'cad_rhd'].values[0]
        By = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_B'), 'cad_rhd'].values[0]
        Bz = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_B'), 'cad_rhd'].values[0]
        Cx = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_C'), 'cad_rhd'].values[0]
        Cy = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_C'), 'cad_rhd'].values[0]
        Cz = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_C'), 'cad_rhd'].values[0]
        Ax_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_A'), 'RHD'].values[0]
        Ay_p= input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_A'), 'RHD'].values[0]
        Az_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_A'), 'RHD'].values[0]
        Bx_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_B'), 'RHD'].values[0]
        By_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_B'), 'RHD'].values[0]
        Bz_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_B'), 'RHD'].values[0]
        Cx_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'x_C'), 'RHD'].values[0]
        Cy_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'y_C'), 'RHD'].values[0]
        Cz_p = input_df.loc[(input_df['Processed_Fragment_ROI'] == roi) & (input_df['Processed_Fragment'] == 'z_C'), 'RHD'].values[0]

        output_data.append({
            'Processed_Fragment_ROI': roi,
            'Ax_p':Ax_p,
            'Ax': Ax,
            'Ay_p':Ay_p,
            'Ay': Ay,
            'Az_p':Az_p,
            'Az': Az,
            'Bx_p':Bx_p,
            'Bx': Bx,
            'By_p':By_p,
            'By': By,
            'Bz_p':Bz_p,
            'Bz': Bz,
            'Cx_p':Cx_p,
            'Cx': Cx,
            'Cy_p':Cy_p,
            'Cy': Cy,
            'Cz_p':Cz_p,
            'Cz': Cz
            
            })

    return pd.DataFrame(output_data)

def process_fragment(fragment):
    # Example: Extract the first character and the second character
    first_char = fragment[0]
    second_char = fragment[2]
    return f'{first_char}_{second_char}'

def process_fragment_last(fragment):
    # Example: Extract the first character and the second character
    #first_char = fragment[0]
    last_digit = fragment.split()[-1]
    return last_digit

def calculate_mae_distances(df):
    # Calculate T and U
    T = df['HP_Y_Pred'] - df['Point_y']
    U = df['HP_Z_Pred'] - df['Point_z']

    # Convert angles to radians and calculate Angle Factor_Y
    Yaw_Pred_rad = np.radians(df['Yaw_Pred'].abs())
    Yaw_Pred_Error_rad = np.radians(df['Yaw_Pred'].abs() + df['Yaw_MAE'].abs())
    Angle_Factor_Y = (np.tan(Yaw_Pred_Error_rad) / np.tan(Yaw_Pred_rad)) - 1

    # Calculate MAE Distance_Y
    df['MAE Distance_Y'] = (T * Angle_Factor_Y).abs()
    # Apply condition based on ROI values
    df.loc[df['ROI_x'].isin(['17_Offroad_Passenger_Exterior', '15_Offroad_Driver_Exterior']), 'MAE Distance_Y'] *= 0.15

    # Convert angles to radians and calculate Angle Factor_P
    Pitch_Pred_rad = np.radians(df['Pitch_Pred'].abs())
    Pitch_Pred_Error_rad = np.radians(df['Pitch_Pred'].abs() + df['Pitch_MAE'].abs())
    Angle_Factor_P = (np.tan(Pitch_Pred_Error_rad) / np.tan(Pitch_Pred_rad)) - 1

    # Calculate MAE Distance_P
    df['MAE Distance_P'] = (U * Angle_Factor_P).abs()

    return df 


def extract_roi_number(roi):
    return int(''.join(filter(str.isdigit, roi)))
    
def calculate_intr_plane(df):
    df.fillna(0, inplace=True)  # Replace null values with 0
    df['Intr_Plane_n1'] = (((df['Ay'] - df['By']) * (df['Az'] - df['Cz'])) - ((df['Ay'] - df['Cy']) * (df['Az'] - df['Bz']))) / 100
    df['Intr_Plane_n2'] = (((df['Ax'] - df['Bx']) * (df['Az'] - df['Cz'])) - ((df['Ax'] - df['Cx']) * (df['Az'] - df['Bz']))) / 100
    df['Intr_Plane_n3'] = (((df['Ax'] - df['Bx']) * (df['Ay'] - df['Cy'])) - ((df['Ax'] - df['Cx']) * (df['Ay'] - df['By']))) / 100
    df['Intr_Plane_d'] = (df['Ax'] * df['Intr_Plane_n1']) + (df['Ay'] * df['Intr_Plane_n2']) + (df['Az'] * df['Intr_Plane_n3'])
    
    # Round off the Intr_Plane values to the nearest whole number
    df['Intr_Plane_n1'] = df['Intr_Plane_n1'].round(0).astype(int)
    df['Intr_Plane_n2'] = df['Intr_Plane_n2'].round(0).astype(int)
    df['Intr_Plane_n3'] = df['Intr_Plane_n3'].round(0).astype(int)
    df['Intr_Plane_d'] = df['Intr_Plane_d'].round(0).astype(int)
    return df
    
def filter_columns(df):
    required_columns = ["ROI", "POI", "Pitch_Pred", "Pitch_MAE", "Yaw_Pred", "Yaw_MAE", "HP_X_Pred", "HP_Y_Pred", "HP_Z_Pred"]
    filtered_df = df[required_columns]
    return filtered_df

def extract_fragment_details(fragment):
    first_char = fragment[0]
    last_digit = fragment.split()[-1]
    return first_char, last_digit

def cad_logic(row):
    lhd = row.get(drive_style, 0)
    rhd = row.get(drive_style, 0)
    co_ordinates = row['Processed_Fragment']
    roi_original = row['Processed_Fragment_ROI']
    mae_distance_y = row.get('MAE Distance_Y', 0)
    mae_distance_p = row.get('MAE Distance_P', 0)
    
    # Ensure all values are numeric
    try:
        lhd = float(lhd)
        rhd = float(rhd)
        mae_distance_y = float(mae_distance_y)
        mae_distance_p = float(mae_distance_p)
    except ValueError:
        lhd = rhd = mae_distance_y = mae_distance_p = 0
    
    if roi_original in ['4','5','9','1','10','6','18','13','11','7','8']:


        if co_ordinates in ['x_A', 'x_B', 'x_C']:
            cad_lhd = lhd
            cad_rhd = rhd
        elif co_ordinates in ['y_A', 'y_C']:
            cad_lhd = lhd - mae_distance_y
            cad_rhd = rhd - mae_distance_y
        elif co_ordinates == 'y_B':
            cad_lhd = lhd + mae_distance_y
            cad_rhd = rhd + mae_distance_y
        elif co_ordinates in ['z_A', 'z_B']:
            cad_lhd = lhd + mae_distance_p
            cad_rhd = rhd + mae_distance_p
        elif co_ordinates == 'z_C':
            cad_lhd = lhd - mae_distance_p
            cad_rhd = rhd - mae_distance_p
        else:
            cad_lhd = lhd
            cad_rhd = rhd

    elif roi_original in ['14','15','16','17']:
            
           

            roi_lhd_A_x = float(roi_dict_lhd[roi_original]['x_A'])
            roi_lhd_B_x = float(roi_dict_lhd[roi_original]['x_B'])
            roi_lhd_C_x = float(roi_dict_lhd[roi_original]['x_C'])

            roi_lhd_A_y = float(roi_dict_lhd[roi_original]['y_A'])
            roi_lhd_B_y = float(roi_dict_lhd[roi_original]['y_B'])
            roi_lhd_C_y = float(roi_dict_lhd[roi_original]['y_C'])

            roi_lhd_A_z = float(roi_dict_lhd[roi_original]['z_A'])
            roi_lhd_B_z = float(roi_dict_lhd[roi_original]['z_B'])
            roi_lhd_C_z = float(roi_dict_lhd[roi_original]['z_C'])


            roi_rhd_A_x = float(roi_dict_rhd[roi_original]['x_A'])
            roi_rhd_B_x = float(roi_dict_rhd[roi_original]['x_B'])
            roi_rhd_C_x = float(roi_dict_rhd[roi_original]['x_C'])

            roi_rhd_A_y = float(roi_dict_rhd[roi_original]['y_A'])
            roi_lhd_B_y = float(roi_dict_rhd[roi_original]['y_B'])
            roi_lhd_C_y = float(roi_dict_rhd[roi_original]['y_C'])

            roi_lhd_A_z = float(roi_dict_rhd[roi_original]['z_A'])
            roi_lhd_B_z = float(roi_dict_rhd[roi_original]['z_B'])
            roi_lhd_C_z = float(roi_dict_rhd[roi_original]['z_C'])
      
            if (roi_lhd_A_x > roi_lhd_B_x ) and (roi_lhd_A_x >= roi_lhd_C_x ) :


                
                #st.write("ROI Dictionary:", roi_dict)
                if co_ordinates in ['y_A', 'y_B', 'y_C']:
                    cad_lhd = lhd
                
                elif co_ordinates in ['x_A','x_C']:
                    cad_lhd = lhd + mae_distance_y
                
                elif co_ordinates in ['x_B']:
                    cad_lhd = lhd - mae_distance_y
                
                elif co_ordinates in ['z_A', 'z_B']:
                    cad_lhd = lhd + mae_distance_p
                
                elif co_ordinates == 'z_C':
                    cad_lhd = lhd - mae_distance_p
                
            else :

                if co_ordinates in ['y_A', 'y_B', 'y_C']:
                    cad_lhd = lhd
                
                elif co_ordinates in ['x_A','x_C']:
                    cad_lhd = lhd - mae_distance_y
                
                elif co_ordinates in ['x_B']:
                    cad_lhd = lhd + mae_distance_y
                
                elif co_ordinates in ['z_A', 'z_B']:
                    cad_lhd = lhd + mae_distance_p
                
                elif co_ordinates == 'z_C':
                    cad_lhd = lhd - mae_distance_p 
                 

            if (roi_rhd_A_x > roi_rhd_B_x ) and (roi_rhd_A_x >= roi_rhd_C_x ) :


                
                #st.write("ROI Dictionary:", roi_dict)
                if co_ordinates in ['y_A', 'y_B', 'y_C']:
                   
                    cad_rhd = rhd
                elif co_ordinates in ['x_A','x_C']:
                   
                    cad_rhd = rhd + mae_distance_y
                elif co_ordinates in ['x_B']:
                  
                    cad_rhd = rhd - mae_distance_y
                elif co_ordinates in ['z_A', 'z_B']:
                
                    cad_rhd = rhd + mae_distance_p
                elif co_ordinates == 'z_C':
                  
                    cad_rhd = rhd - mae_distance_p
                
            else :

                if co_ordinates in ['y_A', 'y_B', 'y_C']:
 
                    cad_rhd = rhd
                elif co_ordinates in ['x_A','x_C']:
               
                    cad_rhd = rhd - mae_distance_y
                elif co_ordinates in ['x_B']:
                
                    cad_rhd = rhd + mae_distance_y
                elif co_ordinates in ['z_A', 'z_B']:
                 
                    cad_rhd = rhd + mae_distance_p
                elif co_ordinates == 'z_C':
                
                    cad_rhd = rhd - mae_distance_p                               

            
    elif roi_original in ['12','20','19']:

        if co_ordinates in ['z_A', 'z_B', 'z_C']:
            cad_lhd = lhd
            cad_rhd = rhd
        elif co_ordinates in ['y_A', 'y_C']:
            cad_lhd = lhd - mae_distance_y
            cad_rhd = rhd - mae_distance_y
        elif co_ordinates == 'y_B':
            cad_lhd = lhd + mae_distance_y
            cad_rhd = rhd + mae_distance_y
        elif co_ordinates in ['x_A', 'x_B']:
            cad_lhd = lhd + mae_distance_p
            cad_rhd = rhd + mae_distance_p
        elif co_ordinates == 'x_C':
            cad_lhd = lhd - mae_distance_p
            cad_rhd = rhd - mae_distance_p
        else:
            cad_lhd = lhd
            cad_rhd = rhd     

               
        
    return pd.Series([cad_lhd, cad_rhd], index=['cad_lhd', 'cad_rhd'])

def add_difference_column(df):
    # Check if the required columns exist in the DataFrame
    if 'Pitch_Pred' in df.columns and 'Yaw_Pred' in df.columns and 'Pitch_GT' in df.columns and 'Yaw_GT' in df.columns:
        # Create new columns 'Pitch_MAE_New' and 'Yaw_MAE_New' as the difference between the respective columns
        df['Pitch_MAE_new'] = df['Pitch_Pred'] - df['Pitch_GT']
        df['Yaw_MAE_new'] = df['Yaw_Pred'] - df['Yaw_GT']
        return df
    else:
        raise ValueError("Required columns not found in the DataFrame.")

def calculate_mean(df, poi_roi_pairs):
    # Filter the main data based on POI and ROI pairs
    filtered_df = df[df.apply(lambda row: (row['ROI_pred'], row['POI']) in poi_roi_pairs, axis=1)]

    filtered_df['Pitch_MAE'] = pd.to_numeric(filtered_df['Pitch_MAE'], errors='coerce')
    filtered_df['Yaw_MAE'] = pd.to_numeric(filtered_df['Yaw_MAE'], errors='coerce')
    filtered_df['Pitch_Pred'] = pd.to_numeric(filtered_df['Pitch_Pred'], errors='coerce')
    filtered_df['Yaw_Pred'] = pd.to_numeric(filtered_df['Yaw_Pred'], errors='coerce')
    filtered_df['HP_X_Pred'] = pd.to_numeric(filtered_df['HP_X_Pred'], errors='coerce')
    filtered_df['HP_Y_Pred'] = pd.to_numeric(filtered_df['HP_Y_Pred'], errors='coerce')
    filtered_df['HP_Z_Pred'] = pd.to_numeric(filtered_df['HP_Z_Pred'], errors='coerce')

    mean_df = filtered_df.groupby(['POI', 'ROI_pred']).agg({
        'Pitch_Pred': 'mean',
        'Pitch_MAE' : 'mean',
        'Yaw_Pred': 'mean',
        'Yaw_MAE': 'mean',
        'HP_X_Pred': 'mean',
        'HP_Y_Pred': 'mean',
        'HP_Z_Pred': 'mean'
    }).reset_index()
    return mean_df


def calculate_mean_given(df, poi_roi_pairs):
    # Filter the main data based on POI and ROI pairs
    filtered_df = df[df.apply(lambda row: (row['ROI_pred'], row['POI']) in poi_roi_pairs, axis=1)]

    filtered_df['Pitch_Pred'] = pd.to_numeric(filtered_df['Pitch_Pred'], errors='coerce')
    filtered_df['Yaw_Pred'] = pd.to_numeric(filtered_df['Yaw_Pred'], errors='coerce')
    filtered_df['HP_X_Pred'] = pd.to_numeric(filtered_df['HP_X_Pred'], errors='coerce')
    filtered_df['HP_Y_Pred'] = pd.to_numeric(filtered_df['HP_Y_Pred'], errors='coerce')
    filtered_df['HP_Z_Pred'] = pd.to_numeric(filtered_df['HP_Z_Pred'], errors='coerce')

    mean_df = filtered_df.groupby(['POI', 'ROI_pred']).agg({
        'Pitch_Pred': 'mean',
        'Yaw_Pred': 'mean',
        'HP_X_Pred': 'mean',
        'HP_Y_Pred': 'mean',
        'HP_Z_Pred': 'mean'
    }).reset_index()
    return mean_df


def calculate_mean_new(df, poi_roi_pairs):
    # Filter the main data based on POI and ROI pairs
    filtered_df = df[df.apply(lambda row: (row['ROI_pred'], row['POI']) in poi_roi_pairs, axis=1)]
    # Group by POI and ROI_pred and calculate mean for the specified columns
    filtered_df['Pitch_MAE_new'] = pd.to_numeric(filtered_df['Pitch_MAE_new'], errors='coerce')
    filtered_df['Yaw_MAE_new'] = pd.to_numeric(filtered_df['Yaw_MAE_new'], errors='coerce')
    filtered_df['Pitch_Pred'] = pd.to_numeric(filtered_df['Pitch_Pred'], errors='coerce')
    filtered_df['Yaw_Pred'] = pd.to_numeric(filtered_df['Yaw_Pred'], errors='coerce')
    filtered_df['HP_X_Pred'] = pd.to_numeric(filtered_df['HP_X_Pred'], errors='coerce')
    filtered_df['HP_Y_Pred'] = pd.to_numeric(filtered_df['HP_Y_Pred'], errors='coerce')
    filtered_df['HP_Z_Pred'] = pd.to_numeric(filtered_df['HP_Z_Pred'], errors='coerce')

    mean_df = filtered_df.groupby(['POI', 'ROI_pred']).agg({
        'Pitch_MAE_new':'mean',
        'Yaw_MAE_new': 'mean',
        'Pitch_Pred': 'mean',
        'Yaw_Pred': 'mean',
        'HP_X_Pred': 'mean',
        'HP_Y_Pred': 'mean',
        'HP_Z_Pred': 'mean'
    }).reset_index()

    mean_df = mean_df.rename(columns={'Pitch_MAE_new': 'Pitch_MAE'})
    mean_df = mean_df.rename(columns={'Yaw_MAE_new': 'Yaw_MAE'})


    return mean_df

def plot_3d_rectangles(df):
    fig = go.Figure()

    # Adding lines to form the rectangle for original points
    for i in range(len(df)):
        roi_label = df['Processed_Fragment_ROI'][i]

        # Line from A to B
        fig.add_trace(go.Scatter3d(x=[df['Ax'][i], df['Bx'][i]], 
                                   y=[df['Ay'][i], df['By'][i]], 
                                   z=[df['Az'][i], df['Bz'][i]],
                                   mode='lines', name='Line A-B',
                                   line=dict(color='blue', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

        # Line from B to C
        fig.add_trace(go.Scatter3d(x=[df['Bx'][i], df['Cx'][i]], 
                                   y=[df['By'][i], df['Cy'][i]], 
                                   z=[df['Bz'][i], df['Cz'][i]],
                                   mode='lines', name='Line B-C',
                                   line=dict(color='blue', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

        # Line from C to D
        df['Dx'] = df['Ax'] + (df['Cx'] - df['Bx'])
        df['Dy'] = df['Ay'] + (df['Cy'] - df['By'])
        df['Dz'] = df['Az'] + (df['Cz'] - df['Bz'])
        fig.add_trace(go.Scatter3d(x=[df['Cx'][i], df['Dx'][i]], 
                                   y=[df['Cy'][i], df['Dy'][i]], 
                                   z=[df['Cz'][i], df['Dz'][i]],
                                   mode='lines', name='Line C-D',
                                   line=dict(color='blue', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

        # Line from D to A
        fig.add_trace(go.Scatter3d(x=[df['Dx'][i], df['Ax'][i]], 
                                   y=[df['Dy'][i], df['Ay'][i]], 
                                   z=[df['Dz'][i], df['Az'][i]],
                                   mode='lines', name='Line D-A',
                                   line=dict(color='blue', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

    # Adding lines to form the rectangle for points with _p suffix
    for i in range(len(df)):
        roi_label = df['Processed_Fragment_ROI'][i]

        # Line from A_p to B_p
        fig.add_trace(go.Scatter3d(x=[df['Ax_p'][i], df['Bx_p'][i]], 
                                   y=[df['Ay_p'][i], df['By_p'][i]], 
                                   z=[df['Az_p'][i], df['Bz_p'][i]],
                                   mode='lines', name='Line A_p-B_p',
                                   line=dict(color='green', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

        # Line from B_p to C_p
        fig.add_trace(go.Scatter3d(x=[df['Bx_p'][i], df['Cx_p'][i]], 
                                   y=[df['By_p'][i], df['Cy_p'][i]], 
                                   z=[df['Bz_p'][i], df['Cz_p'][i]],
                                   mode='lines', name='Line B_p-C_p',
                                   line=dict(color='green', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

        # Line from C_p to D_p
        df['Dx_p'] = df['Ax_p'] + (df['Cx_p'] - df['Bx_p'])
        df['Dy_p'] = df['Ay_p'] + (df['Cy_p'] - df['By_p'])
        df['Dz_p'] = df['Az_p'] + (df['Cz_p'] - df['Bz_p'])
        fig.add_trace(go.Scatter3d(x=[df['Cx_p'][i], df['Dx_p'][i]], 
                                   y=[df['Cy_p'][i], df['Dy_p'][i]], 
                                   z=[df['Cz_p'][i], df['Dz_p'][i]],
                                   mode='lines', name='Line C_p-D_p',
                                   line=dict(color='green', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

        # Line from D_p to A_p
        fig.add_trace(go.Scatter3d(x=[df['Dx_p'][i], df['Ax_p'][i]], 
                                   y=[df['Dy_p'][i], df['Ay_p'][i]], 
                                   z=[df['Dz_p'][i], df['Az_p'][i]],
                                   mode='lines', name='Line D_p-A_p',
                                   line=dict(color='green', width=2),
                                   text=f"ROI: {roi_label}",
                                   hoverinfo='text'))

    # Update the layout for better visualization
    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'),
        width=800,
        margin=dict(r=20, l=10, b=10, t=10))

    st.plotly_chart(fig)

def custom_title():
    custom_title_html = """
    <style>
    .custom-title {
        font-family:'Calibri', sans-serif; /* Change font */
        font-size: 36px; /* Font size */
        color: black; /* Title color */
        text-align: center; /* Center the title */
        margin-top: 20px; /* Space above the title */
        margin-bottom: 10px; /* Space below the title */
        font-weight: bold; /* Make the font bold */
    }
    .separator {
        width: 50%; /* Width of the line */
        height: 2px; /* Thickness of the line */
        background-color: black; /* Color of the line */
        margin: 0 auto; /* Center the line */
        margin-bottom: 20px; /* Space below the line */
    }
    </style>
    <div class="custom-title">POI Based FineTuning</div>
    <div class="separator"></div>
    """
    st.markdown(custom_title_html, unsafe_allow_html=True)

#-------------------------------------------------------function defination end-------------------------------------------------------



#-------------------------------------------------------Main program starts-----------------------------------------------------------
# Initialize transformation object

custom_title()

# Upload files once
pair_file = st.sidebar.file_uploader("Upload the POI and ROI pairs file (CSV or XLSX)", type=["csv", "xlsx"])
file1_upload = st.sidebar.file_uploader("Upload the CAD Input file (CSV or XLSX)", type=["csv", "xlsx"])
uploaded_file = st.sidebar.file_uploader("Upload your main CSV or Excel file", type=["csv", "xlsx"])
drive_style = st.radio("Choose Drive Style", ("LHD", "RHD"))
MAE_input = st.sidebar.radio("Choose MAE_input", ("mean_from_sheet", "mean_given","mean_calculate"))
Car_ID = st.selectbox(
    'Choose an option',
    ('X118-596', 'X253-3059', 'X253-3067','W213-6360','V177-8992','V177-8175')
)


# Ensure all files are uploaded before proceeding
if pair_file  is not None and file1_upload is not None and uploaded_file is not None:

        if st.button("Start Process"):
          
          start_time = time.time()
          
          stStyle = drive_style
          
          mpic_transp_obj = MPIC_Data_Transformations.Transformations(0)
          f = open(r'mbw_data_253.mbw')
          mbw_data = json.load(f)
          f.close()
          mpic_transp_obj.initialize(worldmodel_data=mbw_data)
          poi_roi_pairs = []
         
     
          if pair_file is not None:
              pairs_df = load_data(pair_file)
              if pairs_df is not None and 'POI' in pairs_df.columns and 'ROI' in pairs_df.columns:
                  poi_roi_pairs = list(zip(pairs_df['ROI'], pairs_df['POI']))
                 
              else:
                  st.error("The file must contain 'POI' and 'ROI' columns.")
           
          # Process main data file if uploaded and pairs are provided
          if uploaded_file is not None and poi_roi_pairs:
              df = load_data(uploaded_file)
              file1_df = load_data(file1_upload)
         
              if df is not None:
                  
                  df_file3 = round_numerical_columns(file1_df)

                  df_file3['Processed_Fragment'] = df_file3['Fragment'].apply(process_fragment)
                  df_file3['Processed_Fragment_ROI'] = df_file3['Fragment'].apply(process_fragment_last)
                  df_file3['Processed_Fragment_ROI'] = pd.to_numeric(df_file3['Processed_Fragment_ROI'], errors='coerce')
                  df_file3 = process_csv_roi(df_file3)

                  df_file3 = df_file3.apply(pd.to_numeric, errors='coerce')

                #   st.write("df_file3 roi")
                #   st.write(df_file3)

                 
                 
                  # Calculate mean values

                #   mean_df = add_difference_column(df)

                  if(MAE_input == 'mean_from_sheet'):
                     final_df = calculate_mean(df, poi_roi_pairs)
                     st.write("Mean calculated when the MAE given in the sheet")
                     st.write(final_df)

                  if(MAE_input == 'mean_calculate'):
                     mean_df = add_difference_column(df)
                     final_df = calculate_mean_new(mean_df, poi_roi_pairs)
                     st.write("Mean calculated when the MAE given in the algo sheet")
                     st.write(final_df)

  
                  if(MAE_input == 'mean_given'):
                     st.write("iam inside")
                     mean_df = calculate_mean_given(df, poi_roi_pairs)
                     final_df = pd.merge(mean_df, pairs_df, left_on='POI', right_on='POI', how='left')
                     final_df = final_df.rename(columns={'Pitch_MAE_x': 'Pitch_MAE'})
                     final_df = final_df.rename(columns={'Yaw_MAE_x': 'Yaw_MAE'})

                     st.write(final_df)
                     st.write("Mean calculted when the mean is given with poi and roi pairs")

                     st.write("before mpic to vcs")
                     st.write(final_df)

                  output_data = []
                  for idx, row in final_df.iterrows():
                      VCS_HO_HP = camToVcsHO_HP(row['HP_X_Pred'], row['HP_Y_Pred'], row['HP_Z_Pred'], 0, 0, 0, drive_style, Car_ID,mbw_data)
                      VcsHeadPos_X = VCS_HO_HP["MPIC_D_Head_Pos_X_ST3"]
                      VcsHeadPos_Y = VCS_HO_HP["MPIC_D_Head_Pos_Y_ST3"]
                      VcsHeadPos_Z = VCS_HO_HP["MPIC_D_Head_Pos_Z_ST3"]
                      vcsEyegazeYaw, vcsEyegazePitch = camToVcsGaze(row['Yaw_Pred'], row['Pitch_Pred'], drive_style, Car_ID,mbw_data)
                      output_data.append({
                          "ROI": row['ROI_pred'],
                          "POI": row['POI'],
                          "Pitch_MAE": row['Pitch_MAE'],
                          "Yaw_MAE": row['Yaw_MAE'],
                          "HP_X_Pred": VcsHeadPos_X,
                          "HP_Y_Pred": VcsHeadPos_Y,
                          "HP_Z_Pred": VcsHeadPos_Z,
                          "Yaw_Pred": vcsEyegazeYaw,
                          "Pitch_Pred": vcsEyegazePitch
                      })
                    
                  # Convert list of dictionaries to a pandas DataFrame
                  output_df = pd.DataFrame(output_data)

                  st.write("After mpic to vcs")
                  st.write("output_df")
                  st.write(output_df)

                  output_df = filter_columns(output_df)
                  st.write("output_df")
                  st.write(output_df)

                  df_file3 = calculate_intr_plane(df_file3)
                  st.write("df_file3")
                  st.write(df_file3)

                  pairs_df['ROI_number'] = pairs_df['ROI'].apply(extract_roi_number)
             
                  st.write("pairs_df")
                  st.write(pairs_df)

                #   st.write("output_df")
                #   st.write(output_df)
                 

                  merged_df = pairs_df.merge(output_df, on='POI', how='left')

                #   st.write("after merging ")
                #   st.write(merged_df)



                  st.write("after merginng with pairs file ")
                  st.write(merged_df)
             
                  merged_df = merged_df.merge(df_file3, left_on='ROI_number', right_on='ROI', how='left')
               
         
                  constants = df_file3.set_index('ROI').filter(like='Intr_Plane', axis=1).to_dict(orient='index')
                  merged_df['Intr_Plane_n1'].fillna(merged_df['ROI'].map(lambda roi: constants.get(roi, {}).get('Intr_Plane_n1', np.nan)), inplace=True)
                  merged_df['Intr_Plane_n2'].fillna(merged_df['ROI'].map(lambda roi: constants.get(roi, {}).get('Intr_Plane_n2', np.nan)), inplace=True)
                  merged_df['Intr_Plane_n3'].fillna(merged_df['ROI'].map(lambda roi: constants.get(roi, {}).get('Intr_Plane_n3', np.nan)), inplace=True)
                  merged_df['Intr_Plane_d'].fillna(merged_df['ROI'].map(lambda roi: constants.get(roi, {}).get('Intr_Plane_d', np.nan)), inplace=True)
           
                  st.write("merged_df")
                  st.write(merged_df)

         
                  merged_df = calculate_points(merged_df)
                  st.write("after point calculation")
                  st.write(merged_df)




                  merged_df = merged_df.rename(columns={'Pitch_MAE_x': 'Pitch_MAE'})
                  merged_df = merged_df.rename(columns={'Yaw_MAE_x': 'Yaw_MAE'})


                  merged_df = calculate_mae_distances(merged_df)
                #   st.write("after mae distance ")
                #   st.write(merged_df)

                  
         
         
                  POI_error_df = merged_df[['ROI_number', 'ROI_x', 'POI', 'MAE Distance_P','MAE Distance_Y','Pitch_MAE','Yaw_MAE']]
         
                #   POI_error_df = POI_error_df.sort_values(by ='ROI_number')

                  POI_error_df['MAE Distance_Y'] = POI_error_df['MAE Distance_Y'].round(decimals=0)
                  POI_error_df['MAE Distance_P'] = POI_error_df['MAE Distance_P'].round(decimals=0)

 
                  st.write("Error file with POI's ")
                  st.write(POI_error_df)
         
         
           
                  merged_df = merged_df.groupby('ROI_x')[['MAE Distance_Y', 'MAE Distance_P']].mean().reset_index()
         
                  #merged_df = merged_df.sort_values()
               
                  #st.write(merged_df)
                  # Apply rounding to the final columns before displaying or saving
                  columns_to_round = ['MAE Distance_Y', 'MAE Distance_P']  # Add other columns you want to round
                  merged_df[columns_to_round] = merged_df[columns_to_round].round(0).astype(int)
                  st.write(merged_df)
         
                  ##-------------------CAD Calculation Part--------------------##
         
                  st.write("CAD Calculation Begins")
                  file1_df = round_numerical_columns(file1_df)
         
                  cad_df = merge_data(file1_df,merged_df)
    
                  st.write("new column created???")
                  st.write(cad_df)
    
         
                  cad_df['Processed_Fragment'] = cad_df['Fragment'].apply(process_fragment)
                  cad_df['Processed_Fragment_ROI'] = cad_df['Fragment'].apply(process_fragment_last)
                     
         
         
         
                  roi_dict_lhd = create_roi_dict_lhd(cad_df)
                  roi_dict_rhd = create_roi_dict_rhd(cad_df)
         
                  cad_df[['cad_lhd', 'cad_rhd']] = cad_df.apply(cad_logic, axis=1)
         
               
               
                  cad_df = cad_df[['Processed_Fragment_ROI','ROI_x','Processed_Fragment','LHD','cad_lhd','RHD','cad_rhd','MAE Distance_Y','MAE Distance_P']]
                  cad_df['Processed_Fragment_ROI'] = pd.to_numeric(cad_df['Processed_Fragment_ROI'], errors='coerce')
                  cad_df = cad_df.sort_values(by='Processed_Fragment_ROI')
               
     
     
                  # Define the custom order
                  custom_order = ['x_A', 'y_A', 'z_A', 'x_B', 'y_B', 'z_B', 'x_C', 'y_C', 'z_C']
                  # Convert 'Processed_Fragment' to a categorical type with the specified order
                  cad_df['Processed_Fragment'] = pd.Categorical(cad_df['Processed_Fragment'], categories=custom_order, ordered=True)
                 
                  # Sort the DataFrame by 'Processed_Fragment' and then by 'Processed_Fragment_ROI'
                  sorted_df = cad_df.sort_values(by=['Processed_Fragment_ROI', 'Processed_Fragment'])
                 
                  # Display the sorted DataFrame
                  st.write("new sorted cad wala data")
                  st.write(sorted_df)
     
     
         
                  # POI_err_df = cad_df[['Fragment', 'LHD','cad_lhd','RHD','cad_rhd','ROI_x','MAE Distance_Y', 'MAE Distance_P','Processed_Fragment','Processed_Fragment_ROI']]
                  #merged_df = calculate_point_d(merged_df)
         
                  cad_df = process_csv(cad_df)
           
                 
             
     
                  cad_df['Processed_Fragment_ROI'] = pd.to_numeric(cad_df['Processed_Fragment_ROI'], errors='coerce')
     
                  sorted_cad = cad_df.sort_values(by ='Processed_Fragment_ROI')
     
                  st.write("before calculationg the point d")
                  st.write(sorted_cad)
               
                  cad_df = calculate_point_d(cad_df)
                  

                  cad_df = calculate_point_dx_p(cad_df)    
                  #st.dataframe(cad_df)
     
                 
                  cad_df['Processed_Fragment_ROI'] = pd.to_numeric(cad_df['Processed_Fragment_ROI'], errors='coerce')
     
     
     
                  sorted_final_merge = cad_df.sort_values(by='Processed_Fragment_ROI')
     
                  st.write("sorted merged final")
                  st.write(sorted_final_merge)
         
                  plot_3d_rectangles(cad_df)

                  plot_3d_rectangles()

     
                 
                  #Option to download the merged dataframe as a CSV file
                  st.download_button(
                      label="Download Merged Data as CSV",
                      data=cad_df.to_csv(index=False),
                      file_name="merged_data.csv",
                      mime="text/csv"
                  )

                  end_time = time.time()

                  elapsed_time = end_time - start_time  # Calculate the elapsed time

                  st.success(f"Processing complete! Time taken: {elapsed_time:.2f} seconds")

    
    