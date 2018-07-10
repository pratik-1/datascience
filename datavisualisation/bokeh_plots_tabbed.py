from _datetime import datetime
import time
import pandas as pd
import numpy as np
from bokeh.layouts import column,row,gridplot
from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,CategoricalColorMapper
from bokeh.models.widgets import Tabs,Panel
#from bokeh.models import HoverTool



print('*********************************************')
print('Program "bokeh_plots_tabbed" \nStart Time: ', time.strftime("%Y-%b-%d %H:%M:%S"),'\n\n')
dt1 = datetime.now()


literacy_birth_rate = 'literacy_birth_rate.csv'
iris_dataset = 'Iris dataset/Iris_data'


def get_dataframe(filename):
    """
    :param filename:Data File 
    :return: DataFrame
    """
    df = pd.read_csv(filename,sep=',', header=0)
    # print(df.head())
    return df


def draw_plots_layouts(df):
    """
    Grid layout plots with Box and Lasso select tools
    """
    source = ColumnDataSource(df)
    color_mapper = CategoricalColorMapper(factors=['Iris-setosa','Iris-versicolor','Iris-virginica'],
                                          palette=['red', 'blue', 'green'])

    p1 = figure(x_axis_label='sepal length', y_axis_label='sepal width',tools='box_select,lasso_select')
    p1.circle('sepal length', 'sepal width', size=8, source=source, legend='class',color=dict(field='class', transform = color_mapper)
              )

    p2 = figure(x_axis_label='petal length',y_axis_label='petal width',tools='box_select,lasso_select')
    p2.circle('petal length', 'petal width', size=8, source=source, legend='class',color=dict(field='class', transform = color_mapper)
              )
    p2.legend.location = 'bottom_right'
    p3 = figure(x_axis_label='sepal length', y_axis_label='petal length',tools='box_select,lasso_select')
    p3.circle('sepal length', 'petal length', size=8, source=source, legend='class',
              color=dict(field='class', transform=color_mapper)
              )
    p3.legend.location = 'bottom_right'
    p4 = figure(x_axis_label='sepal width', y_axis_label='petal width',tools='box_select,lasso_select')
    p4.circle('sepal width', 'petal width', size=8, source=source, legend='class',
              color=dict(field='class', transform=color_mapper))
    p4.legend.location = 'center_right'


    # Grid Layout
    row1 = [p1,p2]
    row2 = [p3,p4]
    layout = gridplot([row1, row2])

    # Add the hover tool to the figure p
    ### To-DO ###


    output_file('gridplot_select_tools.html')
    show(layout)


def draw_plots_tabbed(df):
    """
    Tabbed plots
    """
    source = ColumnDataSource(df)
    color_mapper = CategoricalColorMapper(factors=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                                          palette=['red', 'blue', 'green'])

    p1 = figure(x_axis_label='sepal length', y_axis_label='sepal width')
    p1.circle('sepal length', 'sepal width', size=8, source=source, legend='class',
              color=dict(field='class', transform=color_mapper))

    p2 = figure(x_axis_label='petal length', y_axis_label='petal width')
    p2.circle('petal length', 'petal width', size=8, source=source, legend='class',
              color=dict(field='class', transform=color_mapper))
    p2.legend.location = 'bottom_right'
    p3 = figure(x_axis_label='sepal length', y_axis_label='petal length')
    p3.circle('sepal length', 'petal length', size=8, source=source, legend='class',
              color=dict(field='class', transform=color_mapper))
    p3.legend.location = 'bottom_right'
    p4 = figure(x_axis_label='sepal width', y_axis_label='petal width')
    p4.circle('sepal width', 'petal width', size=8, source=source, legend='class',
              color=dict(field='class', transform=color_mapper))
    p4.legend.location = 'center_right'
    first = Panel(child=row(p1,p2),title='First_tab')
    second = Panel(child=row(p3,p4),title='Second_tab')

    tab1 = Tabs(tabs=[first,second])
    output_file('tabbed_plots.html')
    show(tab1)


# dataframe= get_dataframe(iris_dataset)
draw_plots_layouts(get_dataframe(iris_dataset))
draw_plots_tabbed(get_dataframe(iris_dataset))



# *******************************************************
dt2 = datetime.now()

print('\n\nProgram End Time: ', dt2)
print('Time Elapse : ', dt2 - dt1)
print('*****************E   N   D*******************')
