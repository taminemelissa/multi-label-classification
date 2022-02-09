import pandas as pd
from math import *
#fonction qui transforme le fichier parquet en un fichier csv avec des colonnes pour chaque coordonnées des vecteurs usage_features et audio_features
#return en plus le dataframe.csv

def transform_parquet_to_csv(file_path,name_csv):
    df=pd.read_parquet(file_path, engine="pyarrow")
    ind=df.index
    columns1=[]
    for i in range (1,257):
      col1= 'audio_feature_%i' % i
      columns1.append(col1)
    columns2=[]
    for i in range (1,129):
      col2= 'usage_feature_%i' % i
      columns2.append(col2)
    split_df_audio = pd.DataFrame(df['audio_features'].tolist(),columns=columns1,index=ind)
    split_df_usage=pd.DataFrame(df['usage_features'].tolist(),columns=columns2,index=ind)
    df1 = pd.concat([df, split_df_audio], axis=1,index=ind)
    df =pd.concat([df1, split_df_usage], axis=1,index=ind)
    df.drop(columns=['audio_features','usage_features'], inplace=True)
    df.set_index('song_index', inplace=True,drop=True)
    df.to_csv(name_csv)
    return df


#eturn une matrice X : avec les variables explicatives pour chaque musique i (audio+usage ou audio) 
#et une matrice Y: pour chaque musique i, la colonne l = 1 si la musique i appartient au label l 
#p désigne la proportion de données initiales avec laquelle on travaille 
#pour le rendu p=1, pour tester le code p=0.5 
#par défaut on garde que les audio features 
#on choisit d'enlever 'song title' et 'artist_name' pour l'instant

def extract_values_array(df,p,usage=False):
    n=df.shape[0]
    df_extracted= df.sample(int(p*n), random_state=0)
    df_extracted.drop(columns=['song_title','artist_name'],inplace=True)
    #pour qu'on ait des sorties similaires à travers plusieurs appels
    columns3=[]
    if (usage==False):
        for i in range (1,129):
            col2= 'usage_feature_%i' % i
            columns3.append(col2)
        df_extracted.drop(columns=columns3,inplace=True)
    l=list(df_extracted.columns[0:22])
    x=list(df_extracted.columns[22:])
    l.remove('song_index')
    X=df_extracted[x].values
    Y=df_extracted[l].values
    return(X,Y)
