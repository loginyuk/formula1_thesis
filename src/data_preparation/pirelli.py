import numpy as np

def add_pirelli_indices(df):
    """
    Calculate Pirelli-based features based on track and compound characteristics
    """
    df['Compound_Hard_Int'] = df['Compound_Hard_Hardness']
    df['Compound_Medium_Int'] = df['Compound_Medium_Hardness']
    df['Compound_Soft_Int'] = df['Compound_Soft_Hardness']

    # composite features
    df['Wear_Severity_Index'] = df['Asphalt_Abrasion_1_5'] * df['Tyre_Stress_1_5']
    df['Track_Flow_Type']  = df['Lateral_1_5'] / (df['Traction_1_5'] + df['Braking_1_5'])
    df['Grip_Aero_Balance'] = df['Asphalt_Grip_1_5'] / df['Downforce_1_5']
    df['Total_Min_Pressure'] = df['Min_Pressure_Front_PSI'] + df['Min_Pressure_Rear_PSI']
    df['Pressure_Delta'] = df['Min_Pressure_Front_PSI'] - df['Min_Pressure_Rear_PSI']
    return df


def add_compound_features(df):
    """
    Encode compound hardness
    """
    df['Compound_Int'] = np.where(
        df['Compound'] == 'SOFT',
        df['Compound_Soft_Int'],
        np.where(
            df['Compound'] == 'MEDIUM',
            df['Compound_Medium_Int'],
            df['Compound_Hard_Int']))
    
    df['Tyre_Compound_Interaction'] = df['TyreLife'] * df['Compound_Int']
    return df


def build_pirelli_features(df):
    df = add_pirelli_indices(df)
    df = add_compound_features(df)
    return df