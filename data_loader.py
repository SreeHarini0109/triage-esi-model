import os
import pandas as pd
import numpy as np

def calculate_mews(row):
    """
    Calculate Modified Early Warning Score (MEWS) from a row of vital signs.
    """
    score = 0
    
    # Heart Rate
    hr = row.get('HR')
    if pd.notna(hr):
        if hr <= 40: score += 2
        elif 41 <= hr <= 50: score += 1
        elif 51 <= hr <= 100: score += 0
        elif 101 <= hr <= 110: score += 1
        elif 111 <= hr <= 129: score += 2
        elif hr >= 130: score += 3
        
    # Systolic BP
    sbp = row.get('SBP')
    if pd.notna(sbp):
        if sbp <= 70: score += 3
        elif 71 <= sbp <= 80: score += 2
        elif 81 <= sbp <= 100: score += 1
        elif 101 <= sbp <= 199: score += 0
        elif sbp >= 200: score += 2
        
    # Respiratory Rate
    resp = row.get('Resp')
    if pd.notna(resp):
        if resp < 9: score += 2
        elif 9 <= resp <= 14: score += 0
        elif 15 <= resp <= 20: score += 1
        elif 21 <= resp <= 29: score += 2
        elif resp >= 30: score += 3
        
    # Temperature
    temp = row.get('Temp')
    if pd.notna(temp):
        if temp < 35: score += 2
        elif 35 <= temp <= 38.4: score += 0
        elif temp >= 38.5: score += 2
        
    # Exclude AVPU/Consciousness as we only have vitals
    return score

def determine_esi(row):
    """
    Determine proxy ESI level (1=Highest, 5=Lowest) using MEWS, Shock Index, and Vitals.
    """
    mews = row.get('MEWS', 0)
    shock_index = row.get('ShockIndex', 0)
    o2 = row.get('O2Sat', 100)
    
    if pd.isna(o2): 
        o2 = 100

    # ESI 1: Resuscitation
    if mews >= 6 or shock_index > 1.2 or o2 < 85:
        return 1
    # ESI 2: Emergent
    elif mews >= 4 or shock_index > 1.0 or o2 < 90:
        return 2
    # ESI 3: Urgent
    elif mews >= 2 or shock_index > 0.8 or o2 < 95:
        return 3
    # ESI 4: Less Urgent
    elif mews == 1:
        return 4
    # ESI 5: Non-Urgent
    else:
        return 5

def process_patient_file(filepath):
    """
    Load a single patient file (.psv), extract the FIRST valid set of vital signs,
    and calculate derived metrics.
    """
    try:
        df = pd.read_csv(filepath, sep='|')
        if df.empty:
            return None
        
        # We want the FIRST non-null measurement of key features ideally.
        # But for absolute presentation time, we should just take the first row that has at least some vitals.
        # Let's take the first row, but forward fill / back fill lightly if strictly the first row is completely empty.
        # A more rigorous 'presentation time' is the first hour measurement.
        
        # Look at the first row
        first_row = df.iloc[0].copy()
        
        # Vital signs columns
        vitals_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
        
        # If the very first row has completely missing vitals, let's take the first non-null for each column within the first 6 hours
        first_6_hours = df.head(6)
        
        presentation_data = {}
        for col in vitals_cols:
            valid_vals = first_6_hours[col].dropna()
            if not valid_vals.empty:
                presentation_data[col] = valid_vals.iloc[0]
            else:
                presentation_data[col] = np.nan
                
        # Other demographic info is usually static
        presentation_data['Age'] = df['Age'].iloc[0] if 'Age' in df.columns else np.nan
        presentation_data['Gender'] = df['Gender'].iloc[0] if 'Gender' in df.columns else np.nan
        
        patient_id = os.path.basename(filepath).replace('.psv', '')
        presentation_data['PatientID'] = patient_id
        
        # Convert to Series for easy passing
        p_series = pd.Series(presentation_data)
        
        # Calculate Shock Index = HR / SBP
        if pd.notna(p_series['HR']) and pd.notna(p_series['SBP']) and p_series['SBP'] > 0:
            p_series['ShockIndex'] = p_series['HR'] / p_series['SBP']
        else:
            p_series['ShockIndex'] = np.nan
            
        p_series['MEWS'] = calculate_mews(p_series)
        p_series['ESI'] = determine_esi(p_series)
        
        return p_series
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    base_dir = "/Users/sreehariniganishkaa/Downloads/training"
    sets = ['training_setA', 'training_setB']
    
    all_data = []
    
    for s in sets:
        dir_path = os.path.join(base_dir, s)
        if not os.path.exists(dir_path):
            print(f"Skipping {dir_path}, not found.")
            continue
            
        print(f"Processing directory: {s}")
        files = [f for f in os.listdir(dir_path) if f.endswith('.psv')]
        
        for i, file in enumerate(files):
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(files)} files...")
                
            filepath = os.path.join(dir_path, file)
            res = process_patient_file(filepath)
            if res is not None:
                all_data.append(res)
                
    if not all_data:
        print("No data processed!")
        return
        
    final_df = pd.DataFrame(all_data)
    print(f"Total records processed: {len(final_df)}")
    
    # Save to triagemodel dir
    out_path = "/Users/sreehariniganishkaa/triagemodel/presentation_vitals.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")
    
    # Print label distribution
    print("\nESI Label Distribution:")
    print(final_df['ESI'].value_counts().sort_index())

if __name__ == "__main__":
    main()
