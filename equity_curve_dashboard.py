import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Set your ticker symbol here
ticker = 'ETH'  # Replace with the actual ticker, e.g., 'AAPL'

# Load the CSV file
df = pd.read_csv('backtest_equity_comparison.csv', index_col='Date')
df.index = pd.to_datetime(df.index)  # Convert index to datetime if necessary

# Extract windows from column names
equity_cols = [col for col in df.columns if col.startswith('Total_equity_window_')]
windows = sorted([int(col.split('_')[-1]) for col in equity_cols])

app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1('Backtest Visualization'),
    
    html.Div([
        html.Div([
            # Radio for chart type
            html.H3('Select Chart Type:'),
            dcc.RadioItems(
                id='chart-type',
                options=[
                    {'label': 'Line Chart', 'value': 'line'},
                    {'label': 'Monthly Bar Chart', 'value': 'bar'}
                ],
                value='line',
                inline=True
            ),
            
            # Checkboxes for equity curves and signals
            html.H3('Select Equity Curves and Signals:'),
            dcc.Checklist(
                id='equity-signals-checklist',
                options=[
                    {'label': f'Equity window {w}', 'value': f'equity_{w}'} for w in windows
                ] + [
                    {'label': f'Signals window {w}', 'value': f'signals_{w}'} for w in windows
                ],
                value=[f'equity_{w}' for w in windows],  # Initially show all equity curves
                inline=False
            ),
        ], style={'width': '20%', 'padding': '10px'}),
        
        dcc.Graph(id='backtest-graph', style={'height': '800px', 'width': '80%'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
])

# Callback to update the graph based on checklist selections and chart type
@app.callback(
    Output('backtest-graph', 'figure'),
    [Input('equity-signals-checklist', 'value'),
     Input('chart-type', 'value')]
)
def update_graph(selected_values, chart_type):
    if chart_type == 'line':
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.6, 0.4],
                            subplot_titles=(f'{ticker} Price', 'Equity Curves with Trading Signals'))

        # Add price trace (always visible)
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker], name=f'{ticker} Price', line=dict(color='blue'), showlegend=True), row=1, col=1)

        # Add selected traces
        for w in windows:
            equity_col = f'Total_equity_window_{w}'
            show_equity = f'equity_{w}' in selected_values
            show_signals = f'signals_{w}' in selected_values

            # Equity curve
            fig.add_trace(go.Scatter(x=df.index, y=df[equity_col], name=f'Equity window {w}',
                                     visible=show_equity, showlegend=show_equity), row=2, col=1)

            if show_signals:
                open_col = f'{ticker}_open_signal_window_{w}'
                close_col = f'{ticker}_close_signal_window_{w}'
                legendgroup = f'Signals window {w}'
                first_signal = True

                # Open long (>0)
                mask = df[open_col] > 0
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df.loc[mask, equity_col], mode='markers',
                                             marker=dict(symbol='triangle-up', color='green', size=12),
                                             name=f'Signals window {w}' if first_signal else None,
                                             showlegend=first_signal, legendgroup=legendgroup,
                                             visible=True), row=2, col=1)
                    first_signal = False

                # Open short (<0)
                mask = df[open_col] < 0
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df.loc[mask, equity_col], mode='markers',
                                             marker=dict(symbol='triangle-down', color='red', size=12),
                                             name=f'Signals window {w}' if first_signal else None,
                                             showlegend=first_signal, legendgroup=legendgroup,
                                             visible=True), row=2, col=1)
                    first_signal = False

                # Close long (>0)
                mask = df[close_col] > 0
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df.loc[mask, equity_col], mode='markers',
                                             marker=dict(symbol='circle', color='darkgreen', size=10),
                                             name=f'Signals window {w}' if first_signal else None,
                                             showlegend=first_signal, legendgroup=legendgroup,
                                             visible=True), row=2, col=1)
                    first_signal = False

                # Close short (<0)
                mask = df[close_col] < 0
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df.loc[mask, equity_col], mode='markers',
                                             marker=dict(symbol='circle', color='darkred', size=10),
                                             name=f'Signals window {w}' if first_signal else None,
                                             showlegend=first_signal, legendgroup=legendgroup,
                                             visible=True), row=2, col=1)
                    first_signal = False
    else:  # 'bar'
        df_monthly = df.resample('M').last()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.6, 0.4],
                            subplot_titles=(f'{ticker} Monthly Price', 'Monthly Returns (%)'))

        # Add monthly price trace (line)
        fig.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly[ticker], name=f'{ticker} Price', line=dict(color='blue'), showlegend=True), row=1, col=1)

        # Add bar traces for selected equities (monthly returns, no signals)
        selected_windows = [w for w in windows if f'equity_{w}' in selected_values]
        colors = ['rgb(31,119,180)', 'rgb(44,160,44)', 'rgb(214,39,40)', 'rgb(148,103,189)', 'rgb(140,86,75)']  # Example colors for different windows
        
        for i, w in enumerate(selected_windows):
            equity_col = f'Total_equity_window_{w}'
            monthly_equity = df_monthly[equity_col]
            monthly_returns = monthly_equity.pct_change().fillna(0) * 100
            fig.add_trace(go.Bar(x=df_monthly.index, y=monthly_returns, name=f'Returns window {w}',
                                 marker_color=colors[i % len(colors)]), row=2, col=1)

        fig.update_layout(barmode='group')

    # Common layout updates
    fig.update_layout(
        hovermode='x unified',  # Unified hover for better crosshair experience
        spikedistance=1000,     # Distance for spikes to appear
        height=800, width=1200, title_text='Backtest Visualization',
        legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.05)
    )
    
    # Enable spikelines (crosshair) on both axes
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='black', spikedash='solid', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikecolor='black', spikedash='solid', spikethickness=1)

    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8057)