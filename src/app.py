from flask import Flask, render_template, request
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import random
from shapely.geometry import Polygon
import shapely.geometry

app = Flask(__name__)

# Variables that effect output
mice = 5
speed = 0.001
steps = 1000000
stretch_ratio = 1
random_mice_outline = False
random_mice_spacing = False

# Finding angle for equidistant mice
spacing = np.linspace((2 * math.pi) / mice, 2 * math.pi, mice)
# Defining the x and y values and distance as blank lists to append to
mousex = []
mousey = []

# Making mice equidistant by unit circle trig
for mouse in spacing:
    mousex.append(math.sin(mouse))
    mousey.append(math.cos(mouse))

def drawoutline(xs, ys, mice, fig):

    dict = {'xs': xs, 'ys': ys}
    df = pd.DataFrame(dict)
    fig.add_scatter(df)

    coordpairs = list((zip(xs, ys)))
    polygon = Polygon(coordpairs)
    centre = polygon.centroid
    print(centre)


def start_ngrok():
        from pyngrok import ngrok
        url = ngrok.connect(5000)
        print('Tunnel URL:', url)


def get_mouse_graph(mice, random_mice_spacing, stretch_ratio):
    offset = 0
    if mice % 2 == 0:
        offset = -(math.pi / mice)
    # Finding angle for equidistant mice
    if random_mice_outline:
        spacing = [random.uniform(0, 2 * math.pi) for x in range(mice)]
        spacing.sort()
    else:
        spacing = np.linspace((2 * math.pi) / mice, 2 * math.pi, mice)
        spacing = [x + offset for x in spacing]
    # Defining the x and y values and distance as blank lists to append to
    mousex = []
    mousey = []
    distx = []
    disty = []
    mousenx = np.zeros((mice, steps))
    mouseny = np.zeros((mice, steps))
    areas = np.zeros((steps))
    # Making mice equidistant by unit circle trig
    # Changing shape dimensions
    if random_mice_spacing:
        for mouse in spacing:
            stretch_ratiox = random.uniform(0.1, 1)
            stretch_ratioy = random.uniform(0.1, 1)
            mousex.append(math.sin(mouse) * stretch_ratiox * stretch_ratio)
            mousey.append(math.cos(mouse) * stretch_ratioy)
            distx.append(0)
            disty.append(0)
    else:
        for mouse in spacing:
            mousex.append(math.sin(mouse) * stretch_ratio)
            mousey.append(math.cos(mouse))
            distx.append(0)
            disty.append(0)
    # print(mousex,mousey)
    fig = go.Figure(layout=go.Layout(template="simple_white"))
    #ax = fig.add_subplot(111)
    #ax.set_aspect("equal")
    firstx = mousex.copy()
    firsty = mousey.copy()
    firstx.append(firstx[0])
    firsty.append(firsty[0])
    # 2D array of points that make up convex hull
    convex_hull = np.array(
        shapely.geometry.MultiPoint(
            [xy for xy in zip(firstx, firsty)]
        ).convex_hull.exterior.coords
    )


    polygon1 = go.Scatter(
        x=convex_hull[:, 0],
        y=convex_hull[:, 1],
        showlegend=False,
        mode="lines",
        fill="none",
        line=dict(color="LightSeaGreen", width=2),
    )

    fig.add_trace(polygon1)


    #drawoutline(firstx, firsty, mice, fig)

    # Plotting original points on graph
    # plt.scatter(mousex,mousey,color="black",marker=".")
    # Working out when to stop
    caught = np.zeros(mice)
    pathlength = 0
    for i in range(steps):
        if sum(caught) == mice:
            pathlength = i * speed
            mousenx = mousenx[:, :i]
            mouseny = mouseny[:, :i]
            print("Each path has length", pathlength)
            break
        for mouse in range(mice):
            # Finding mouse distance for normal case
            if mouse == mice - 1:
                distx[mouse] = (mousex[0] - mousex[mouse])
                disty[mouse] = (mousey[0] - mousey[mouse])
                # Finding mouse distance for last case
            else:
                distx[mouse] = (mousex[mouse + 1] - mousex[mouse])
                disty[mouse] = (mousey[mouse + 1] - mousey[mouse])
        # Finding unit vector to move mice by constant distance
        for mouse in range(mice):
            moddistance = math.sqrt(distx[mouse] ** 2 + disty[mouse] ** 2)
            unitdistance = speed / moddistance
            # Stopping the mice going through each other
            if unitdistance > moddistance:
                caught[mouse] = True
            # Moving the mice
            movex = distx[mouse] * unitdistance
            movey = disty[mouse] * unitdistance
            mousenx[mouse, i] = mousex[mouse]
            mouseny[mouse, i] = mousey[mouse]
            mousex[mouse] = mousex[mouse] + movex
            mousey[mouse] = mousey[mouse] + movey
            mousenx[mouse, i + 1] = mousex[mouse]
            mouseny[mouse, i + 1] = mousey[mouse]
    #fig = px.line(x=mousenx[0,:], y=mouseny[0,:])
    for i in range(mice):
        fig.add_scatter(x=mousenx[i,:], y=mouseny[i,:])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(showlegend=False)
    graphJSON = fig.to_json()
    return json.dumps(graphJSON)


def get_graph(period='JJA', mice=3):
    df = pd.read_csv('../GLB.Ts+dSST.csv')
    fig = px.bar(df, x='Year', y=period,
                 color=period, title=period,
                 color_continuous_scale='reds',
                 template='plotly_white', width=1000, height=500)

    graphJSON = fig.to_json()
    return json.dumps(graphJSON)


def template(params):
    return render_template(params['template'], params=params)

@app.route('/', methods=['POST', 'GET'])
def simpleindex():
    header = "Mice Problem Simulation"
    subheader = "{} mice, {} type, {} stretch".format(5, 'Regular', 1)
    description = """Set the parameters below and click submit to replot. 
    """
    params = {
        'template': 'index.html',
        'title': header,
        'subtitle': subheader,
        'content': description,
        'mice': mice,
        'random_mice_spacing': random_mice_spacing,
        'stretch': stretch_ratio,
        'type': "regular",
        'graph': get_mouse_graph(mice, False, 1)

    }

    if request.method == 'POST':
        params["mice"] = request.form['mice']
        params["stretch"] = request.form['stretch']
        if (request.form['type']) == "Random":
            params["random_mice_spacing"] = True
            params["type"] = 'Random'
        if (request.form['type']) == "Regular":
            params["random_mice_spacing"] = False
            params["type"] = "Regular"
        if (request.form['type']) == "No Stretch":
            params["stretch"] = "1"
        params['subtitle'] = "{} mice, {} type, {} stretch".format(params["mice"], params["type"], params["stretch"])
        params['graph'] = get_mouse_graph(int(request.form['mice']), params["random_mice_spacing"],
                                          float(params["stretch"]))

        return template(params)

    return template(params)



if __name__ == '__main__':
    app.run()
    start_ngrok()
