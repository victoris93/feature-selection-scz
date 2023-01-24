import sys
import numpy as np 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pingouin import intraclass_corr

def gradDispersion(grads_ses1, grads_ses2, n_neighbours, save = False, filename = None):

    vertex_nbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(grads_ses1)
    distances, indices = vertex_nbrs.kneighbors(grads_ses1)
    vd_ses1 = distances.mean(axis = 1)

    vertex_nbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(grads_ses2)
    distances, indices = vertex_nbrs.kneighbors(grads_ses2)
    vd_ses2 = distances.mean(axis = 1)
        
    vd = np.asarray([vd_ses1, vd_ses2])
    if save == True:
        np.save(arr = vd, file = filename)
    return vd

def make_measure_df(measure_array, measure_name, subject, session, save = False, filename = None):
    measure_df = pd.DataFrame(measure_array, columns = [measure_name])
    measure_df["Subject"] = subject
    measure_df["Session"] = session
    measure_df["Vertex"] = np.arange(len(measure_df))
    if save == True:
        measure_df.to_csv(f"{filename}")
    return measure_df
     
def vertexICC(measure_df, ICC_type = "ICC2"):
    vertices = np.arange(len(measure_df[0]))
    vertex_wise_ICC = []
    for vertex in vertices:
        data = measure_df[measure_df["Vertex"] == vertex]
        ICC = intraclass_corr(data, targets = "Subject", raters = "Session", ratings = "Dispersion")
        ICC["Vertex"] = vertex
        vertex_wise_ICC.append(ICC[ICC["Type"] == ICC_type])
    vertex_wise_ICC = pd.concat(vertex_wise_ICC)
    return vertex_wise_ICC