import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import isaacgym
from isaacgymenvs.tasks.dexterity.env.tool_utils import DexterityCategory
import torch

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Load the shape space
shape_space = DexterityCategory(
    gym=None,
    sim=None,
    demo_path=None,
    asset_root=None,
    source_file=None,
    target_files=[None, ]
)
shape_space.from_file(load_path='/home/user/mosbach/PycharmProjects/dexterity/isaacgymenvs/tasks/dexterity/demo/category_space/drill_category', load_instances=False)


# ------------------------------------------------------------------------------
# Define control components.

switches = html.Div(
    [
        dbc.Label("Visualization options"),
        dbc.Checklist(
            options=[
                {"label": "Show keypoints", "value": 1},
                {"label": "Show manipulator mesh", "value": 2},
            ],
            value=[1],
            id="switches-input",
            switch=True,
        ),
    ]
)

dropdown = html.Div(
    [
        dbc.Label("Train instance to show"),
        dcc.Dropdown(shape_space.latent_space_df['name'], id='train-instance-dropdown'),
    ]
)


latent_space_params = html.Div(
    [
            html.H5("Latent (shape) space parameters:"),
            #html.P("First principal component:"),
            dcc.Input(id='first_principal_comp', value=0.0, type="number", style={'marginRight':'10px', 'width': 100}),
            #html.P("Second principal component:"),
            dcc.Input(id='second_principal_comp', value=0.0, type="number", style={'width': 100}),
    ]
)

buttons = html.Div(
    [
        html.H5("Commands"),
        dbc.Button("Set latent params", id="update", color="primary",
                   style={"margin": "5px"}, n_clicks_timestamp='0'),
        dbc.Button("Fit joints to keypoints ", id="fit_joints",
                   color="secondary", style={"margin": "5px"},
                   n_clicks_timestamp='0'),
    ]
)

latent_shape_space_scatter = html.Div(
    [
        html.H5("Latent (shape) space"),
        dcc.Graph(id='latent_space',)
    ]
)

deformed_pointcloud_scatter = html.Div(
    [
        dcc.Graph(animate=True,
                  animation_options={ 'frame': { 'False': True, }, 'transition': { 'duration': 750, 'ease': 'cubic-in-out', },}, id='deformed_pointcloud'),
        dcc.Interval(
            id='interval-component',
            interval=2*1000, # in milliseconds
            n_intervals=0
        )
    ]
)


controls = [
        latent_space_params,
        buttons,
        switches,
        dropdown,
        latent_shape_space_scatter
    ]


app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.H2("Grasp Pose Generalization"),
            ], width=True),
        ], align="end"),

        html.Hr(),

        dbc.Row(
            [
                dbc.Col(controls, width=3),
                dbc.Col([deformed_pointcloud_scatter, ], width=True)
            ]
        )
    ],
    fluid=True
)


n_clicks = {
    'update': 0,
    'fit_joints': 0,
    'intervals': 0
}

meshes_visible = None
meshes_invisible = None
last_switches_input = [1]

@app.callback(
    [
        Output('latent_space', 'figure'),
        Output('deformed_pointcloud', 'figure'),
        Output('first_principal_comp', 'value'),
        Output('second_principal_comp', 'value'),
     ],
    [
        Input('update', 'n_clicks'),
        Input('fit_joints', 'n_clicks'),
        Input("switches-input", "value"),
        Input("train-instance-dropdown", "value"),
        Input("latent_space", "clickData"),

    ],
    [
        State('first_principal_comp', 'value'),
        State('second_principal_comp', 'value'),
    ]
)
def display_geometry(
        update_n_clicks,
        fit_joints_n_clicks,
        switches_input,
        train_instance_dd_input,
        click_data,
        first_principal_comp,
        second_principal_comp,
):
    # Check button and switch inputs
    update_n_clicks = update_n_clicks or 0
    fit_joints_n_clicks = fit_joints_n_clicks or 0

    update_clicked = update_n_clicks > n_clicks['update']
    fit_joints_clicked = fit_joints_n_clicks > n_clicks['fit_joints']
    n_clicks['update'] += int(update_clicked)
    n_clicks['fit_joints'] += int(fit_joints_clicked)
    show_keypoints = 1 in switches_input
    show_manipulator_mesh = 2 in switches_input

    # Query latent space position from click_data if it caused the update.
    if update_clicked:
        pass
    elif click_data:
        first_principal_comp = click_data['points'][0]['x']
        second_principal_comp = click_data['points'][0]['y']

    latent_space_params = torch.zeros(shape_space.num_latents)
    latent_space_params[0] = first_principal_comp
    latent_space_params[1] = second_principal_comp

    ls_figure = shape_space.draw_latent_space(
        torch.Tensor([first_principal_comp, second_principal_comp]),
        show=False)

    if train_instance_dd_input is not None:
        show_training_instances = [train_instance_dd_input, ]
    else:
        show_training_instances = []

    dpc_figure = shape_space.draw_deformed_pointcloud(
        latent_space_params, show_keypoints=show_keypoints,
        show_training_instances=show_training_instances)

    global meshes_visible, meshes_invisible
    # Fit meshes to current drill.
    if fit_joints_clicked:
        meshes_visible, meshes_invisible = shape_space.draw_manipulator_mesh(
            latent_space_params)
    # Initialize manipulator meshes.
    elif meshes_visible is None:
        meshes_visible, meshes_invisible = shape_space.draw_manipulator_mesh(
            torch.zeros(shape_space.num_latents))

    if show_manipulator_mesh:
        for mesh3d in meshes_visible:
            dpc_figure.add_trace(mesh3d)
    else:
        for mesh3d in meshes_invisible:
            dpc_figure.add_trace(mesh3d)

    return [ls_figure, dpc_figure, first_principal_comp, second_principal_comp]


try:
    app.title = "Grasp pose generalization"
except:
    print("Could not set the page title!")
app.run_server(debug=True)
