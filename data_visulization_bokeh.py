# Use bokeh as a data visualization tool
bokeh.sampledata.download()

#==============================================================================
# #example 1: mutiple data in one figure
#==============================================================================
from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure, show, output_file

colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
flowers['color'] = flowers['species'].map(lambda x: colormap[x])

output_file("iris.html", title="iris.py example")

p = figure(title = "Iris Morphology")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'

p.circle(flowers["petal_length"], flowers["petal_width"],
        color=flowers["color"], fill_alpha=0.2, size=10, )

show(p)
#==============================================================================
#  Stack of multiple sinsoidal wave
#==============================================================================
import numpy as np

from bokeh.plotting import figure, show, output_file, vplot

N = 100

x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)

output_file("legend.html", title="legend.py example")

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

p1 = figure(title="Legend Example", tools=TOOLS)

p1.circle(x, y, legend="sin(x)")
p1.circle(x, 2*y, legend="2*sin(x)", color="orange", )
p1.circle(x, 3*y, legend="3*sin(x)", color="green", )

p2 = figure(title="Another Legend Example", tools=TOOLS)

p2.circle(x, y, legend="sin(x)")
p2.line(x, y, legend="sin(x)")

p2.line(x, 2*y, legend="2*sin(x)",
    line_dash=[4, 4], line_color="orange", line_width=2)

p2.square(x, 3*y, legend="3*sin(x)", fill_color=None, line_color="green")
p2.line(x, 3*y, legend="3*sin(x)", line_color="green")

show(vplot(p1, p2))  # open a browser


#==============================================================================
# plot correlations
#==============================================================================
import time

from numpy import cumprod, linspace, random

from bokeh.plotting import figure, show, output_file, vplot

num_points = 300

now = time.time()
dt = 24*3600 # days in seconds
dates = linspace(now, now + num_points*dt, num_points) * 1000 # times in ms
acme = cumprod(random.lognormal(0.0, 0.04, size=num_points))
choam = cumprod(random.lognormal(0.0, 0.04, size=num_points))

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

output_file("correlation.html", title="correlation.py example")

r = figure(x_axis_type = "datetime", tools=TOOLS)

r.line(dates, acme, color='#1F78B4', legend='ACME')
r.line(dates, choam, color='#FB9A99', legend='CHOAM')

r.title = "Stock Returns"
r.grid.grid_line_alpha=0.3

c = figure(tools=TOOLS)

c.circle(acme, choam, color='#A6CEE3', legend='close')

c.title = "ACME / CHOAM Correlations"
c.grid.grid_line_alpha=0.3

show(vplot(r, c))  # open a browser

#==============================================================================
# glucose in range
#==============================================================================

import pandas as pd

from bokeh.sampledata.glucose import data
from bokeh.plotting import figure, show, output_file, vplot

output_file("glucose.html", title="glucose.py example")

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p1 = figure(x_axis_type="datetime", tools=TOOLS)

p1.line(data.index, data['glucose'], color='red', legend='glucose')
p1.line(data.index, data['isig'], color='blue', legend='isig')

p1.title = "Glucose Measurements"
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Value'

day = data.ix['2010-10-06']
highs = day[day['glucose'] > 180]
lows = day[day['glucose'] < 80]

p2 = figure(x_axis_type="datetime", tools=TOOLS)

p2.line(day.index.to_series(), day['glucose'],
    line_color="gray", line_dash="4 4", line_width=1, legend="glucose")
p2.circle(highs.index, highs['glucose'], size=6, color='tomato', legend="high")
p2.circle(lows.index, lows['glucose'], size=6, color='navy', legend="low")

p2.title = "Glucose Range"
p2.xgrid[0].grid_line_color=None
p2.ygrid[0].grid_line_alpha=0.5
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Value'

data['inrange'] = (data['glucose'] < 180) & (data['glucose'] > 80)
window = 30.5*288 #288 is average number of samples in a month
inrange = pd.rolling_sum(data.inrange, window)
inrange = inrange.dropna()
inrange = inrange/float(window)

p3 = figure(x_axis_type="datetime", tools=TOOLS)

p3.line(inrange.index, inrange, line_color="navy")

p3.title = "Glucose In-Range Rolling Sum"
p3.xaxis.axis_label = 'Date'
p3.yaxis.axis_label = 'Proportion In-Range'

show(vplot(p1,p2,p3))

#==============================================================================
# confusion matrix
#==============================================================================
from collections import OrderedDict

import numpy as np

from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.sampledata.les_mis import data

nodes = data['nodes']
names = [node['name'] for node in sorted(data['nodes'], key=lambda x: x['group'])]

N = len(nodes)
counts = np.zeros((N, N))
for link in data['links']:
    counts[link['source'], link['target']] = link['value']
    counts[link['target'], link['source']] = link['value']

colormap = [
    "#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"
]

xname = []
yname = []
color = []
alpha = []
for i, n1 in enumerate(nodes):
    for j, n2 in enumerate(nodes):
        xname.append(n1['name'])
        yname.append(n2['name'])

        a = min(counts[i,j]/4.0, 0.9) + 0.1
        alpha.append(a)

        if n1['group'] == n2['group']:
            color.append(colormap[n1['group']])
        else:
            color.append('lightgrey')


source = ColumnDataSource(
    data=dict(
        xname=xname,
        yname=yname,
        colors=color,
        alphas=alpha,
        count=counts.flatten(),
    )
)

output_file("les_mis.html")

p = figure(title="Les Mis Occurrences",
    x_axis_location="above", tools="resize,hover,save",
    x_range=list(reversed(names)), y_range=names)
p.plot_width = 800
p.plot_height = 800

p.rect('xname', 'yname', 0.9, 0.9, source=source,
     color='colors', alpha='alphas', line_color=None)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3

hover = p.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([
    ('names', '@yname, @xname'),
    ('count', '@count'),
])

show(p)      # show the plot

#==============================================================================
# heat map for unemployment
#==============================================================================
from collections import OrderedDict

import numpy as np

from bokeh.plotting import ColumnDataSource, figure, show, output_file
from bokeh.models import HoverTool
from bokeh.sampledata.unemployment1948 import data

# Read in the data with pandas. Convert the year column to string
data['Year'] = [str(x) for x in data['Year']]
years = list(data['Year'])
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
data = data.set_index('Year')

# this is the colormap from the original plot
colors = [
    "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce",
    "#ddb7b1", "#cc7878", "#933b41", "#550b1d"
]

# Set up the data for plotting. We will need to have values for every
# pair of year/month names. Map the rate to a color.
month = []
year = []
color = []
rate = []
for y in years:
    for m in months:
        month.append(m)
        year.append(y)
        monthly_rate = data[m][y]
        rate.append(monthly_rate)
        color.append(colors[min(int(monthly_rate)-2, 8)])

source = ColumnDataSource(
    data=dict(month=month, year=year, color=color, rate=rate)
)

output_file('unemployment.html')

TOOLS = "resize,hover,save,pan,box_zoom,wheel_zoom"

p = figure(title="US Unemployment (1948 - 2013)",
    x_range=years, y_range=list(reversed(months)),
    x_axis_location="above", plot_width=900, plot_height=400,
    toolbar_location="left", tools=TOOLS)

p.rect("year", "month", 1, 1, source=source,
    color="color", line_color=None)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3

hover = p.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([
    ('date', '@month @year'),
    ('rate', '@rate'),
])

show(p)      # show the plot

#==============================================================================
# multilines in one figure
#==============================================================================
import numpy as np

from bokeh.sampledata.stocks import AAPL, FB, GOOG, IBM, MSFT
from bokeh.plotting import figure, show, output_file, vplot

output_file("stocks.html", title="stocks.py example")

p1 = figure(x_axis_type = "datetime")

p1.line(np.array(AAPL['date'], 'M64'), AAPL['adj_close'], color='#A6CEE3', legend='AAPL')
p1.line(np.array(FB['date'], 'M64'), FB['adj_close'], color='#1F78B4', legend='FB')
p1.line(np.array(GOOG['date'], 'M64'), GOOG['adj_close'], color='#B2DF8A', legend='GOOG')
p1.line(np.array(IBM['date'], 'M64'), IBM['adj_close'], color='#33A02C', legend='IBM')
p1.line(np.array(MSFT['date'], 'M64'), MSFT['adj_close'], color='#FB9A99', legend='MSFT')

p1.title = "Stock Closing Prices"
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'

aapl = np.array(AAPL['adj_close'])
aapl_dates = np.array(AAPL['date'], dtype=np.datetime64)

window_size = 30
window = np.ones(window_size)/float(window_size)
aapl_avg = np.convolve(aapl, window, 'same')

p2 = figure(x_axis_type="datetime")

p2.circle(aapl_dates, aapl, size=4, color='darkgrey', alpha=0.2, legend='close')
p2.line(aapl_dates, aapl_avg, color='navy', legend='avg')

p2.title = "AAPL One-Month Average"
p2.grid.grid_line_alpha=0
p2.xaxis.axis_label = 'Date'
p2.yaxis.axis_label = 'Price'
p2.ygrid.band_fill_color="olive"
p2.ygrid.band_fill_alpha = 0.1

show(vplot(p1,p2))  # open a browser

#==============================================================================
# category heat map
#==============================================================================
from bokeh.plotting import figure, show, output_file, vplot

N = 4000

factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
x0 = [0, 0, 0, 0, 0, 0, 0, 0]
x =  [50, 40, 65, 10, 25, 37, 80, 60]

output_file("categorical.html", title="categorical.py example")

p1 = figure(title="Dot Plot", tools="resize,save", y_range=factors, x_range=[0,100])

p1.segment(x0, factors, x, factors, line_width=2, line_color="green", )
p1.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

factors = ["foo", "bar", "baz"]
x = ["foo", "foo", "foo", "bar", "bar", "bar", "baz", "baz", "baz"]
y = ["foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"]
colors = [
    "#0B486B", "#79BD9A", "#CFF09E",
    "#79BD9A", "#0B486B", "#79BD9A",
    "#CFF09E", "#79BD9A", "#0B486B"
]

p2 = figure(title="Categorical Heatmap", tools="resize,hover,save",
    x_range=factors, y_range=factors)

p2.rect(x, y, color=colors, width=1, height=1)

show(vplot(p1, p2))  # open a browser

#==============================================================================
# histogram
#==============================================================================
import numpy as np
import scipy.special

from bokeh.plotting import figure, show, output_file, vplot

output_file('histogram.html')

p1 = figure(title="Normal Distribution (μ=0, σ=0.5)",tools="save",
       background_fill="#E8DDCB")

mu, sigma = 0, 0.5

measured = np.random.normal(mu, sigma, 1000)
hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(-2, 2, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
     fill_color="#036564", line_color="#033649",\
)
p1.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p1.line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

p1.legend.orientation = "top_left"
p1.xaxis.axis_label = 'x'
p1.yaxis.axis_label = 'Pr(x)'



p2 = figure(title="Log Normal Distribution (μ=0, σ=0.5)", tools="save",
       background_fill="#E8DDCB")

mu, sigma = 0, 0.5

measured = np.random.lognormal(mu, sigma, 1000)
hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(0, 8.0, 1000)
pdf = 1/(x* sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))
cdf = (1+scipy.special.erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))/2

p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    fill_color="#036564", line_color="#033649")
p2.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p2.line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

p2.legend.orientation = "bottom_right"
p2.xaxis.axis_label = 'x'
p2.yaxis.axis_label = 'Pr(x)'



p3 = figure(title="Gamma Distribution (k=1, θ=2)", tools="save",
       background_fill="#E8DDCB")

k, theta = 1.0, 2.0

measured = np.random.gamma(k, theta, 1000)
hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(0, 20.0, 1000)
pdf = x**(k-1) * np.exp(-x/theta) / (theta**k * scipy.special.gamma(k))
cdf = scipy.special.gammainc(k, x/theta) / scipy.special.gamma(k)

p3.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    fill_color="#036564", line_color="#033649")
p3.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p3.line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

p3.legend.orientation = "top_left"
p3.xaxis.axis_label = 'x'
p3.yaxis.axis_label = 'Pr(x)'



p4 = figure(title="Beta Distribution (α=2, β=2)", tools="save",
       background_fill="#E8DDCB")

alpha, beta = 2.0, 2.0

measured = np.random.beta(alpha, beta, 1000)
hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(0, 1, 1000)
pdf = x**(alpha-1) * (1-x)**(beta-1) / scipy.special.beta(alpha, beta)
cdf = scipy.special.btdtr(alpha, beta, x)

p4.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    fill_color="#036564", line_color="#033649")
p4.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p4.line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

p4.xaxis.axis_label = 'x'
p4.yaxis.axis_label = 'Pr(x)'



p5 = figure(title="Weibull Distribution (λ=1, k=1.25)", tools="save",
       background_fill="#E8DDCB")

lam, k = 1, 1.25

measured = lam*(-np.log(np.random.uniform(0, 1, 1000)))**(1/k)
hist, edges = np.histogram(measured, density=True, bins=50)

x = np.linspace(0, 8, 1000)
pdf = (k/lam)*(x/lam)**(k-1) * np.exp(-(x/lam)**k)
cdf = 1 - np.exp(-(x/lam)**k)

p5.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    fill_color="#036564", line_color="#033649")
p5.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p5.line(x, cdf, line_color="white", line_width=2, alpha=0.7, legend="CDF")

p5.legend.orientation = "top_left"
p5.xaxis.axis_label = 'x'
p5.yaxis.axis_label = 'Pr(x)'



show(vplot(p1,p2,p3,p4,p5))

#==============================================================================
# high dimensional data
#==============================================================================
from collections import OrderedDict
from math import log, sqrt

import numpy as np
import pandas as pd
from six.moves import cStringIO as StringIO

from bokeh.plotting import figure, show, output_file

antibiotics = """
bacteria,                        penicillin, streptomycin, neomycin, gram
Mycobacterium tuberculosis,      800,        5,            2,        negative
Salmonella schottmuelleri,       10,         0.8,          0.09,     negative
Proteus vulgaris,                3,          0.1,          0.1,      negative
Klebsiella pneumoniae,           850,        1.2,          1,        negative
Brucella abortus,                1,          2,            0.02,     negative
Pseudomonas aeruginosa,          850,        2,            0.4,      negative
Escherichia coli,                100,        0.4,          0.1,      negative
Salmonella (Eberthella) typhosa, 1,          0.4,          0.008,    negative
Aerobacter aerogenes,            870,        1,            1.6,      negative
Brucella antracis,               0.001,      0.01,         0.007,    positive
Streptococcus fecalis,           1,          1,            0.1,      positive
Staphylococcus aureus,           0.03,       0.03,         0.001,    positive
Staphylococcus albus,            0.007,      0.1,          0.001,    positive
Streptococcus hemolyticus,       0.001,      14,           10,       positive
Streptococcus viridans,          0.005,      10,           40,       positive
Diplococcus pneumoniae,          0.005,      11,           10,       positive
"""

drug_color = OrderedDict([
    ("Penicillin",   "#0d3362"),
    ("Streptomycin", "#c64737"),
    ("Neomycin",     "black"  ),
])

gram_color = {
    "positive" : "#aeaeb8",
    "negative" : "#e69584",
}

df = pd.read_csv(StringIO(antibiotics),
                 skiprows=1,
                 skipinitialspace=True,
                 engine='python')

width = 800
height = 800
inner_radius = 90
outer_radius = 300 - 10

minr = sqrt(log(.001 * 1E4))
maxr = sqrt(log(1000 * 1E4))
a = (outer_radius - inner_radius) / (minr - maxr)
b = inner_radius - a * maxr

def rad(mic):
    return a * np.sqrt(np.log(mic * 1E4)) + b

big_angle = 2.0 * np.pi / (len(df) + 1)
small_angle = big_angle / 7

x = np.zeros(len(df))
y = np.zeros(len(df))

output_file("burtin.html", title="burtin.py example")

p = figure(plot_width=width, plot_height=height, title="",
    x_axis_type=None, y_axis_type=None,
    x_range=[-420, 420], y_range=[-420, 420],
    min_border=0, outline_line_color="black",
    background_fill="#f0e1d2", border_fill="#f0e1d2")

p.line(x+1, y+1, alpha=0)

# annular wedges
angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
colors = [gram_color[gram] for gram in df.gram]
p.annular_wedge(
    x, y, inner_radius, outer_radius, -big_angle+angles, angles, color=colors,
)

# small wedges
p.annular_wedge(x, y, inner_radius, rad(df.penicillin),
    -big_angle+angles+5*small_angle, -big_angle+angles+6*small_angle,
    color=drug_color['Penicillin'])
p.annular_wedge(x, y, inner_radius, rad(df.streptomycin),
    -big_angle+angles+3*small_angle, -big_angle+angles+4*small_angle,
    color=drug_color['Streptomycin'])
p.annular_wedge(x, y, inner_radius, rad(df.neomycin),
    -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
    color=drug_color['Neomycin'])

# circular axes and lables
labels = np.power(10.0, np.arange(-3, 4))
radii = a * np.sqrt(np.log(labels * 1E4)) + b
p.circle(x, y, radius=radii, fill_color=None, line_color="white")
p.text(x[:-1], radii[:-1], [str(r) for r in labels[:-1]],
    text_font_size="8pt", text_align="center", text_baseline="middle")

# radial axes
p.annular_wedge(x, y, inner_radius-10, outer_radius+10,
    -big_angle+angles, -big_angle+angles, color="black")

# bacteria labels
xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))
yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))
label_angle=np.array(-big_angle/2+angles)
label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
p.text(xr, yr, df.bacteria, angle=label_angle,
    text_font_size="9pt", text_align="center", text_baseline="middle")

# OK, these hand drawn legends are pretty clunky, will be improved in future release
p.circle([-40, -40], [-370, -390], color=list(gram_color.values()), radius=5)
p.text([-30, -30], [-370, -390], text=["Gram-" + gr for gr in gram_color.keys()],
    text_font_size="7pt", text_align="left", text_baseline="middle")

p.rect([-40, -40, -40], [18, 0, -18], width=30, height=13,
    color=list(drug_color.values()))
p.text([-15, -15, -15], [18, 0, -18], text=list(drug_color.keys()),
    text_font_size="9pt", text_align="left", text_baseline="middle")

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

show(p)

#==============================================================================
# area stack
#==============================================================================
from collections import OrderedDict

import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.palettes import brewer

N = 20
categories = ['y' + str(x) for x in range(10)]
data = {}
data['x'] = np.arange(N)
for cat in categories:
    data[cat] = np.random.randint(10, 100, size=N)

df = pd.DataFrame(data)
df = df.set_index(['x'])

def stacked(df, categories):
    areas = OrderedDict()
    last = np.zeros(len(df[categories[0]]))
    for cat in categories:
        next = last + df[cat]
        areas[cat] = np.hstack((last[::-1], next))
        last = next
    return areas

areas = stacked(df, categories)

colors = brewer["Spectral"][len(areas)]

x2 = np.hstack((data['x'][::-1], data['x']))

output_file("brewer.html", title="brewer.py example")

p = figure()
p.patches([x2 for a in areas], list(areas.values()), color=colors, alpha=0.8, line_color=None)

p.grid.minor_grid_line_color = '#eeeeee'
show(p)

#==============================================================================
# plot line
#==============================================================================
import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.sampledata import periodic_table

elements = periodic_table.elements
elements = elements[elements["atomic number"] <= 82]
elements = elements[~pd.isnull(elements["melting point"])]
mass = [float(x.strip("[]")) for x in elements["atomic mass"]]
elements["atomic mass"] = mass

palette = list(reversed([
    "#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#92c5de","#4393c3","#2166ac","#053061"
]))

melting_points = elements["melting point"]
low = min(melting_points)
high= max(melting_points)
melting_point_inds = [int(10*(x-low)/(high-low)) for x in melting_points] #gives items in colors a value from 0-10
meltingpointcolors = [palette[i] for i in melting_point_inds]

output_file("elements.html", title="elements.py example")

TOOLS = "pan,wheel_zoom,box_zoom,reset,resize,save"

p = figure(tools=TOOLS, toolbar_location="left", logo="grey", plot_width=1200)
p.title = "Density vs Atomic Weight of Elements (colored by melting point)"
p.background_fill= "#cccccc"

p.circle(elements["atomic mass"], elements["density"], size=12,
       color=meltingpointcolors, line_color="black", fill_alpha=0.8)

p.text(elements["atomic mass"], elements["density"]+0.3,
    text=elements["symbol"],text_color="#333333",
    text_align="center", text_font_size="10pt")

p.xaxis.axis_label="atomic weight (amu)"
p.yaxis.axis_label="density (g/cm^3)"
p.grid.grid_line_color="white"

show(p)

#==============================================================================
# area chart
#==============================================================================
from collections import OrderedDict

from bokeh.charts import Area, show, output_file

# create some example data
xyvalues = OrderedDict(
    python=[2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111],
    pypy=[12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130],
    jython=[22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160],
)

output_file(filename="area.html")

area = Area(
    xyvalues, title="Area Chart",
    xlabel='time', ylabel='memory',
    stacked=True, legend="top_left"
).legend("top_left")

show(area)