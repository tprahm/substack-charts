import pandas as pd
from datetime import datetime, timedelta
import requests
import io
import plotly.graph_objects as go
import plotly.io as pio
import traceback

# Define the base URLs for the CBOE data
base_url = "https://markets.cboe.com/us/futures/market_statistics/settlement/csv?dt="
vix_urls = {
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX6M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
    "VIX1Y": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1Y_History.csv",
}

# Function to fetch CBOE Futures Data
def fetch_cboe_futures(date_offset=0):
    target_date = (datetime.now() - timedelta(days=date_offset)).strftime("%Y-%m-%d")
    url = base_url + target_date
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
        return data, target_date
    except Exception as e:
        print("Error fetching CBOE data:")
        print(traceback.format_exc())
        return None, target_date

# Function to fetch historical index data for the given key
def fetch_vix_eod(index_key):
    url = vix_urls[index_key]
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))

        # Parse the DATE column to match the expected format
        data['DATE'] = pd.to_datetime(data['DATE'], format="%m/%d/%Y")
        yesterday = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")

        yesterday_data = data[data['DATE'] == yesterday]
        if not yesterday_data.empty:
            return yesterday_data.iloc[0]['CLOSE']
        else:
            print(f"No {index_key} data available for {yesterday}")
            return None
    except Exception as e:
        print(f"Error fetching {index_key} index data:")
        print(traceback.format_exc())
        return None

# Function to fetch specific or all VIX indices
def fetch_vix_indices(selected_indices=None):
    """
    Fetch specified VIX indices or all indices if none are specified.
    
    Parameters:
        selected_indices (list or None): List of index keys to fetch. 
                                         Defaults to None, which fetches all indices.
    
    Returns:
        dict: A dictionary with index keys as keys and close values as values.
    """
    if selected_indices is None:
        selected_indices = list(vix_urls.keys())  # Fetch all indices if none specified

    vix_data = {}
    for index_key in selected_indices:
        vix_data[index_key] = fetch_vix_eod(index_key)
    return vix_data

# Main function to fetch data and create the plot
def main(selected_indices_to_plot=None, plot_previous_day=False):
    """
    Main function to fetch data and plot VIX futures and specified indices.
    
    Parameters:
        selected_indices_to_plot (list or None): List of indices to include in the plot.
                                                 Defaults to None, which fetches all.
        plot_previous_day (bool): Whether to plot the previous day's term structure.
    """
    try:
        # Fetch today's data
        data, today = fetch_cboe_futures()
        if data is None:
            print("Failed to fetch CBOE Futures data. Exiting.")
            return

        # Filter VX futures
        vx_futures = data[data['Symbol'].str.match(r'^VX/')]
        vx_futures = vx_futures.loc[vx_futures['Expiration Date'] > today].copy()
        vx_futures['Expiration Date'] = pd.to_datetime(vx_futures['Expiration Date'])
        vx_futures = vx_futures.sort_values('Expiration Date')

        # Fetch previous day's data if required
        previous_vx_futures = None
        if plot_previous_day:
            prev_data, prev_date = fetch_cboe_futures(date_offset=1)
            if prev_data is not None:
                previous_vx_futures = prev_data[prev_data['Symbol'].str.match(r'^VX/')]
                previous_vx_futures = previous_vx_futures.loc[previous_vx_futures['Expiration Date'] > prev_date].copy()
                previous_vx_futures['Expiration Date'] = pd.to_datetime(previous_vx_futures['Expiration Date'])
                previous_vx_futures = previous_vx_futures.sort_values('Expiration Date')

        # Fetch specified VIX indices
        vix_data = fetch_vix_indices(selected_indices_to_plot)

        # Create a Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vx_futures['Expiration Date'],
            y=vx_futures['Price'],
            mode='lines+markers+text',
            marker=dict(size=10, color='#FFA500', symbol='circle'),
            line=dict(width=4, color='#FFA500'),
            text=[f"${price:,.2f}" for price in vx_futures['Price']],
            textposition='bottom center',
            name='VIX Futures (Today)'
        ))

        # Add previous day's term structure
        if plot_previous_day and previous_vx_futures is not None:
            fig.add_trace(go.Scatter(
                x=previous_vx_futures['Expiration Date'],
                y=previous_vx_futures['Price'],
                mode='lines+markers',
                marker=dict(size=10, color='rgba(255,165,0,0.7)', symbol='circle'),
                line=dict(width=4, color='rgba(255,165,0,0.7)'),
                name='VIX Futures (Previous Day)'
            ))

        # Add horizontal lines for selected VIX indices
        if "VIX" in vix_data and vix_data["VIX"] is not None:
            close_value = vix_data["VIX"]
            min_date = vx_futures['Expiration Date'].min()
            max_date = vx_futures['Expiration Date'].max()

            # Add horizontal line for VIX
            fig.add_trace(go.Scatter(
                x=[min_date, max_date],
                y=[close_value, close_value],
                mode='lines',
                line=dict(dash='dot', width=2),
                name="VIX"
            ))

            # Add far-right text label for VIX
            fig.add_trace(go.Scatter(
                x=[max_date],
                y=[close_value],
                mode='text',
                text=[f"${close_value:,.2f}"],
                textposition='middle right',
                showlegend=False
            ))

        fig.update_layout(
            title="VIX Futures & Selected Index Term Structure",
            xaxis_title="Expiration Date",
            yaxis_title="Price",
            yaxis=dict(
                autorange=True,  # Enable autoscaling for the y-axis
                fixedrange=False  # Ensure the range can change dynamically
            ),
            plot_bgcolor='#EDEDED',
            paper_bgcolor='#EDEDED',
            hovermode="x unified",
            font=dict(size=14, color='black')
        )

        # Export to an HTML file
        output_file = "vix_term_structure_selected.html"
        pio.write_html(fig, output_file)
        print(f"Plot saved as {output_file}")

    except Exception as e:
        print("An error occurred in the main function:")
        print(traceback.format_exc())

# Call the main function, specifying only VIX and enabling previous day plotting
if __name__ == "__main__":
    main(selected_indices_to_plot=["VIX"], plot_previous_day=True)