from _datetime import datetime
import time
import pandas as pd
from bkcharts import Histogram,show,output_file,BoxPlot,gridplot,Scatter
# from bokeh.models import HoverTool


print('*********************************************')
print('Program "bokeh_plots" \nStart Time: ', time.strftime("%Y-%b-%d %H:%M:%S"), '\n\n')

dt1 = datetime.now()

# literacy_birth_rate = 'literacy_birth_rate.csv'
iris_dataset = 'Iris dataset/Iris_data'
autodataset = 'automobiles.csv'
fishdataset = 'fish.csv'


def get_dataframe(filename):
    """
    :param filename:Data File 
    :return: DataFrame
    """
    df = pd.read_csv(filename,sep=',', header=0)
    # print(df.head())
    return df


def histograms(df):
    p = Histogram(df, 'sepal length', color='class', label='class', legend='top_right')
    output_file('irisdataset.html')
    show(p)


def boxplot(df):
    p1 = BoxPlot(df, values='mpg', color='origin', label='origin', title='AutoInfo by region', legend='top_right')
    p2 = BoxPlot(df, values='hp', color='origin', label='origin', title='AutoInfo by region', legend='top_right')
    p3 = BoxPlot(df, values='accel', color='origin', label='origin', title='AutoInfo by region', legend='top_right')
    p4 = BoxPlot(df, values='weight', color='origin', label='origin', title='AutoInfo by region', legend='top_right')

    layout = gridplot([p1,p2],[p3,p4])
    output_file('autodataset.html')
    show(layout)


def scatterplot(df):
    s1 = Scatter(df, x='weight', y='height', color='species', marker='species', title='Fish Data', legend='top_right')
    s2 = Scatter(df, x='weight', y='width', color='species', marker='species', title='Fish Data', legend='top_right')
    s3 = Scatter(df, x='weight', y='length1', color='species', marker='species', title='Fish Data', legend='top_right')
    s4 = Scatter(df, x='width', y='height', color='species', marker='species', title='Fish Data', legend='top_right')

    layout = gridplot([s1,s2],[s3,s4])
    output_file('fishdataset.html')
    show(layout)


histograms(get_dataframe(iris_dataset))
boxplot(get_dataframe(autodataset))
scatterplot(get_dataframe(fishdataset))


# *******************************************************
dt2 = datetime.now()

print('\n\nProgram End Time: ', dt2)
print('Time Elapse : ', dt2 - dt1)
print('*****************E   N   D*******************')