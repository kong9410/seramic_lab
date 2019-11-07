from django.shortcuts import render
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.http import JsonResponse

# 회귀 곡선 관련 모듈
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import os
import itertools
import json

# 몽고 관련 라이브러리
from . import mongo_connect
# Create your views here.
def index(request):
    return render(request, 'index.html')

def corr(request):
    return render(request, 'corr.html')

def cthree(request):
    return render(request, 'c3.html')

def get_data(request):
    mg = mongo_connect.mongodb()
    db = mg.set_db('ceramicdb')
    mg.set_collection('ceramic')

    col = mg.get_collection()
    info = []
    result = col.find({})
    for r in result:
        if r["thickness1"] == "측정불가":
            continue
        if r['failure'] == "":
            r['failure'] = 1
            info.append(r)
        elif r['failure'] != "":
            r['failure'] = 2
            info.append(r)

    df = pd.DataFrame(info)
    df = df.corr()
    col = list(df.columns.values)
    data = []
    # 데이터 전처리
    for r in col:
        for c in col:
            temp = {
                "axis1" : r,
                "axis2" : c,
                "value" : df[r][c],
            }
            data.append(temp)
    context = {
        "data" : data,
        "columns" : col,
    }
    return JsonResponse(context)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
# api/corr_data
# c3 data type
def corr_data(request):
    # connect to mongodb
    mc = mongo_connect.mongodb()
    mc.set_db('ceramicdb')
    mc.set_collection('ceramic')
    col = mc.get_collection()
    
    # db query
    result = col.find({})
    # 데이터 프레임으로 전환
    df = pd.DataFrame(result)
    input_name = 'Gap'
    output_name = 'average_thickness'
    # 문자열을 숫자로 바꾸고 NAN 행 제거
    df['average_thickness'] = pd.to_numeric(df['average_thickness'], errors='coerce')
    df = df.dropna(axis = 0)
    index = list(df['index'].values)
    # x, y scatter 값과 회귀 곡선을 그리는 좌표 반환
    x, y, x_fit, y_quad = quadratic_y_values(df, input_name, output_name)
    x_fit = list(itertools.chain(*x_fit))
    x = list(x)
    y = list(y)
    y_quad = list(y_quad)

    scatter_len = len(x)
    fit_len = len(x_fit)

    print(mean_confidence_interval(y_quad))

    for idx in range(scatter_len):
        x[idx] = float(x[idx])
        y[idx] = float(y[idx])
    
    for idx in range(fit_len):
        x_fit[idx] = float(x_fit[idx])
        y_quad[idx] = float(y_quad[idx])

    x_label = "Gap"
    y_label = "Average_Thickness"

    x.insert(0, 'x_data')
    y.insert(0, y_label + '/' + x_label)
    x_fit.insert(0, 'x_fit')
    y_quad.insert(0, 'Regression')


    context = {
        "x_data" : x,
        "y_data" : y,
        "x_label" : x_label,
        "y_label" : y_label,
        "x_fit" : x_fit,
        "y_quad" : y_quad
    }
    # print(context)
    return JsonResponse(context)

# api/scatter
# d3 data type
def scatter_plot(request):
    print("scatter_plot")
    # connect to mongodb
    mc = mongo_connect.mongodb()
    mc.set_db('ceramicdb')
    mc.set_collection('ceramic')
    col = mc.get_collection()

    # query
    result = col.find({})

    # 데이터 프레임으로 전환
    df = pd.DataFrame(result)
    input_name = 'Gap'
    output_name = 'average_thickness'
    # 문자열을 숫자로 바꾸고 NAN 행 제거
    df['average_thickness'] = pd.to_numeric(df['average_thickness'], errors='coerce')
    df = df.dropna(axis = 0)
    index = list(df['index'].values)
    # x, y scatter 값과 회귀 곡선을 그리는 좌표 반환
    x, y, x_fit, y_quad = quadratic_y_values(df, input_name, output_name)
    x_fit = list(itertools.chain(*x_fit))
    x = list(x)

    # D3 데이터 형식으로 넣어줄 수 있도록 형태 변환
    scatter_len = len(x)
    line_len = len(x_fit)
    scatter_data = []
    for idx in range(scatter_len):
        temp = {
            "index" : int(index[idx]),
            "x_data" : int(x[idx]),
            "y_data" : int(y[idx]),
        }
        scatter_data.append(temp)
    
    line_data = []
    for idx in range(line_len):
        temp = {
            "x_fit" : int(x_fit[idx]),
            "y_quad" : int(y_quad[idx]),
        }
        line_data.append(temp)
    context = {
        'scatter_data' : scatter_data,
        'line_data' : line_data,
    }
    print("json sended")
    # Json으로 반환
    return JsonResponse(context)

# make trendline and x, y plot data
def makegraph(first, second):
    first = first
    second = second
    f_mean = 0
    s_mean = 0
    for a in first:
        f_mean = f_mean + a
    for s in second:
        s_mean = s_mean + s
    f_mean = f_mean / len(first)
    s_mean = s_mean / len(second)
    xr = 0
    yr = 0
    term1 = 0
    term2 = 0
    for i in range(len(second)):
        xr = first[i] - f_mean
        yr = second[i] - s_mean
        term1 += xr * yr
        term2 += xr * xr
    b1 = term1 / term2
    b0 = s_mean - (b1 * f_mean)

    yhat = []
    for i in range(len(second)):
        yhat.append(b0 + (first[i] * b1))
    first = list(first)
    second = list(second)
    data = []
    for i in range(len(second)):
        temp = {
            "yhat" : yhat[i],
            "y" : second[i],
            "x" : first[i],
        }
        data.append(temp)
    return data


def quadratic_y_values(df, x_name, y_name, degree=2):
    X = df[[x_name]].values
    Y = df[y_name].values
    lr = LinearRegression()
    quadratic = PolynomialFeatures(degree=degree)
    X_quad = quadratic.fit_transform(X)
    X_fit = np.arange(0, X.max()+1, 100)[:, np.newaxis]
    lr.fit(X_quad, Y)
    y_quad_fit = lr.predict(quadratic.fit_transform(X_fit))
    return df[x_name].values, Y, X_fit, y_quad_fit