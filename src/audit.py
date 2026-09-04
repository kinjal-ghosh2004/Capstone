import os
import pandas as pd
import json

def analyze_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    
    metadata = {
        'File': os.path.basename(filepath),
        'Format': ext,
        'Samples': 0,
        'Columns': '',
        'Text_Column': '',
        'Label_Column': '',
        'Labels': '',
        'Missing_Values': 0,
        'Max_Length': 0,
        'Avg_Length': 0
    }
    
    try:
        if ext in ['.csv', '.tsv']:
            sep = '\t' if ext == '.tsv' else ','
            try:
                df = pd.read_csv(filepath, sep=sep, on_bad_lines='skip', engine='python', encoding='latin-1')
                # If only one column found in csv, it might be tab separated or headerless
                if len(df.columns) <= 1 and ext == '.csv':
                    df = pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip', engine='python', encoding='latin-1')
            except Exception:
                df = pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip', engine='python', encoding='latin-1')
                
            metadata['Samples'] = len(df)
            metadata['Columns'] = ', '.join([str(c) for c in df.columns])
            metadata['Missing_Values'] = df.isna().sum().sum()
            
            text_col = None
            label_col = None
            for col in df.columns:
                if df[col].dtype == object:
                    avg_len = df[col].dropna().astype(str).apply(len).mean()
                    if avg_len > 20:
                        text_col = col
                    elif df[col].nunique() < 20:
                        label_col = col
            
            if not text_col and len(df.columns) > 0: text_col = df.columns[0]
            if not label_col and len(df.columns) > 1: label_col = df.columns[1]
            
            metadata['Text_Column'] = str(text_col)
            metadata['Label_Column'] = str(label_col)
            
            if text_col is not None and text_col in df.columns:
                lens = df[text_col].dropna().astype(str).apply(len)
                metadata['Max_Length'] = lens.max() if len(lens) > 0 else 0
                metadata['Avg_Length'] = round(lens.mean(), 2) if len(lens) > 0 else 0
                
            if label_col is not None and label_col in df.columns:
                metadata['Labels'] = ', '.join([str(x) for x in df[label_col].dropna().unique()][:10])
                
        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            metadata['Samples'] = len(lines)
            if len(lines) > 0:
                try:
                    obj = json.loads(lines[0])
                    metadata['Columns'] = ', '.join(obj.keys())
                    if 'INDIC REVIEW' in obj: 
                        metadata['Text_Column'] = 'INDIC REVIEW'
                    if 'LABEL' in obj: 
                        metadata['Label_Column'] = 'LABEL'
                    
                    labels = set()
                    lens = []
                    for line in lines:
                        try:
                            o = json.loads(line)
                            if 'LABEL' in o: labels.add(str(o['LABEL']))
                            if 'INDIC REVIEW' in o: lens.append(len(str(o['INDIC REVIEW'])))
                        except:
                            pass
                    metadata['Labels'] = ', '.join(list(labels)[:10])
                    if lens:
                        metadata['Max_Length'] = max(lens)
                        metadata['Avg_Length'] = round(sum(lens)/len(lens), 2)
                except:
                    pass
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        
    return metadata

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datasets_dir = os.path.join(base_dir, 'Datasets')
results = []
for root, _, files in os.walk(datasets_dir):
    for f in files:
        if f.endswith(('.csv', '.tsv', '.json')):
            res = analyze_file(os.path.join(root, f))
            res['Directory'] = os.path.relpath(root, datasets_dir)
            results.append(res)

df_res = pd.DataFrame(results)
cols = ['Directory'] + [c for c in df_res.columns if c != 'Directory']
df_res = df_res[cols]
out_dir = os.path.join(base_dir, 'results')
os.makedirs(out_dir, exist_ok=True)
df_res.to_csv(os.path.join(out_dir, 'dataset_audit.csv'), index=False)
print(f"Audit complete. Processed {len(df_res)} files.")

