import os
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustOptionBacktester:
    """
    An extensive backtester that:
      - Iterates EOD option data from CSV(s) in a directory
      - Can open multiple positions in one day (e.g., sell put + sell call)
      - Calculates day-by-day PnL for each open position
      - Produces a day-by-day PnL log where each row can represent one position
        on a given date (i.e., potentially multiple rows per date).
    """
    def __init__(
        self,
        data_directory,
        positions,  # New parameter
        start_date=None,
        end_date=None,
        day_of_entry="Monday",
        monthly_expiration=True,
        contract_size=1,
        opening_fee=0.75,
        closing_fee=0.75,
        exclude_decimal_strikes=False,
        custom_columns=None,
        output_csv=None,
        daily_pnl_output_csv=None,
        closed_positions_output_csv=None,
    ):
        self.data_directory = data_directory
        self.positions = positions  # Store positions
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.day_of_entry = day_of_entry
        self.monthly_expiration = monthly_expiration
        self.contract_size = contract_size
        self.opening_fee = opening_fee
        self.closing_fee = closing_fee
        self.exclude_decimal_strikes = exclude_decimal_strikes

        self.output_csv = output_csv
        self.daily_pnl_output_csv = daily_pnl_output_csv
        self.closed_positions_output_csv = closed_positions_output_csv

        default_columns = {
            "date_col": "c_date",
            "expiration_col": "expiration_date",
            "call_put_col": "call_put",
            "delta_col": "delta",
            "strike_col": "price_strike",
            "ask_col": "Ask",
            "bid_col": "Bid",
            "underlying_price_col": "underlying_price",
            "otm_col": "calc_OTM",
            "option_symbol_col": "option_symbol",
            "openinterest_col": "openinterest",
            "iv_col": "iv",
            "dte_col": "dte",
        }

        self.columns = {**default_columns, **(custom_columns or {})}
        
        # Maintain a list of open positions
        self.open_positions = []
        
        # Initialize dictionaries to track cumulative PnL and cost basis per date
        self.date_pnl_summary = {}       # {date: cumulative_raw_pnl}
        self.date_cost_basis = {}        # {date: cumulative_cost_basis}

        # Initialize CSV writer for daily PnL
        if self.daily_pnl_output_csv:
            with open(self.daily_pnl_output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "date", "status", "symbol", "side", "call_put", "expiration",
                    "strike", "delta", "underlying_price", "pos_price",
                    "quantity", "openinterest", "iv", "total_raw_pnl",
                    "total_pct_pnl", "fees_incurred",
                    "combined_total_raw_pnl", "combined_total_pct_pnl"  # Added combined columns
                ])
                writer.writeheader()

    # ----------------------------------------------------------
    # File Handling
    # ----------------------------------------------------------
    
    def _list_csv_files(self):
        """Return a list of CSV paths from either a directory or single file."""
        if os.path.isfile(self.data_directory):
            return [self.data_directory]
        elif os.path.isdir(self.data_directory):
            files = [
                os.path.join(self.data_directory, f)
                for f in os.listdir(self.data_directory)
                if f.lower().endswith(".csv")
            ]
            files.sort()
            return files
        else:
            raise ValueError(f"Invalid path: {self.data_directory}")

    def _read_data(self, file_path):
        """Read one CSV, convert date columns, filter by start_date/end_date if needed."""
        df = pd.read_csv(file_path)
        df[self.columns["date_col"]] = pd.to_datetime(df[self.columns["date_col"]], errors="coerce")
        df[self.columns["expiration_col"]] = pd.to_datetime(df[self.columns["expiration_col"]], errors="coerce")

        if self.start_date:
            df = df[df[self.columns["date_col"]] >= self.start_date]
        if self.end_date:
            df = df[df[self.columns["date_col"]] <= self.end_date]

        df.sort_values(by=self.columns["date_col"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _is_third_friday(self, dt):
        """
        Identify the third Friday of the month, including cases where expiration is stored as the Saturday after the third Friday.
        """
        # If the date is a Friday in the 15th-21st range
        if (dt.weekday() == 4) and (15 <= dt.day <= 21):
            return True
        
        # If the date is a Saturday, check if the previous day is a valid third Friday
        if (dt.weekday() == 5) and (15 <= (dt - pd.Timedelta(days=1)).day <= 21):
            return True

        return False

    # ----------------------------------------------------------
    # Price Helpers
    # ----------------------------------------------------------
    
    def get_open_price(self, row, trade_side):
        """
        If we're buying, we open at the Ask.
        If we're selling, we open at the Bid.
        """
        if trade_side.lower() == "buy":
            return row[self.columns["ask_col"]]
        else:
            return row[self.columns["bid_col"]]

    def get_close_price(self, row, trade_side):
        """
        If we originally bought (trade_side=buy), we'd close by selling at the Bid.
        If we originally sold (trade_side=sell), we'd close by buying at the Ask.
        """
        if trade_side.lower() == "buy":
            return row[self.columns["bid_col"]]
        else:
            return row[self.columns["ask_col"]]

    def get_mid_price(self, row):
        """Compute the midpoint price from bid/ask."""
        return 0.5 * (row[self.columns["ask_col"]] + row[self.columns["bid_col"]])

    # ----------------------------------------------------------
    # Position Management
    # ----------------------------------------------------------
    
    def open_new_position(self, row, trade_side, quantity, pnl_buffer):
        """
        Create a single new open-position dict for tracking.
        e.g., if user wants to short a Put, trade_side="sell", call_put="P".
        """
        open_price = self.get_open_price(row, trade_side)

        new_pos = {
            "symbol": row[self.columns["option_symbol_col"]],
            "open_datetime": row[self.columns["date_col"]],
            "expiration": row[self.columns["expiration_col"]],
            "call_put": row[self.columns["call_put_col"]],  # 'C' or 'P'
            "strike": row[self.columns["strike_col"]],
            "delta": row[self.columns["delta_col"]],
            "underlying_price_at_expiration": None,         # Placeholder for expiration price
            "trade_side": trade_side.lower(),              # 'buy' or 'sell'
            "pos_value": float(open_price),
            "quantity": quantity,                           # Use quantity from position definition
            "fees_incurred": 0.0,                           # Initialize fees
            "opening_fee_applied": False,
            "closing_fee_applied": False,
        }

        self.open_positions.append(new_pos)
        
        # Log an 'open' record to daily PnL
        self._log_daily_pnl(position=new_pos, row=row, status="open", pnl_buffer=pnl_buffer)

    def open_multiple_positions(self, positions_to_open, daily_df, pnl_buffer):
        """
        Open multiple positions simultaneously based on the provided positions list.

        :param positions_to_open: List of dictionaries, each containing 'side', 'callput', 'quantity', and 'strike_selection'
        :param daily_df: DataFrame containing option data for the current day
        :param pnl_buffer: List to accumulate PnL records for the current date
        """
        for pos in positions_to_open:
            side = pos.get('side', 'sell')  # Default to 'sell' if not specified
            callput = pos.get('callput', 'C')  # Default to 'C' if not specified
            quantity = pos.get('quantity', self.contract_size)  # Default to contract_size
            strike_selection = pos.get('strike_selection', 'ATM')  # Default to 'ATM'

            # Select the appropriate option row based on 'callput' and 'strike_selection'
            row = self.pick_contract(daily_df, callput, strike_selection)

            if row is not None:
                self.open_new_position(row, side, quantity, pnl_buffer)
            else:
                logger.warning(f"No {callput} option found to open position with strike selection: {strike_selection}")

    def close_position(self, position, row_close, pnl_buffer):
        """
        Close a single given open position.
        On expiration day or the adjusted expiration (Friday before),
        calculate intrinsic value instead of using market price.
        """
        current_date = row_close[self.columns["date_col"]].date()
        expiration_date = position["expiration"].date()
        day_before_expiration = expiration_date - pd.Timedelta(days=1)
        
        # Check if the day is the day before expiration and a Friday
        is_adjusted_expiration = current_date == day_before_expiration and day_before_expiration.weekday() == 4

        if current_date >= expiration_date or is_adjusted_expiration:
            # Calculate intrinsic value at expiration
            if position["call_put"] == "C":  # Call option
                intrinsic_value = max(0, row_close[self.columns["underlying_price_col"]] - position["strike"])
            elif position["call_put"] == "P":  # Put option
                intrinsic_value = max(0, position["strike"] - row_close[self.columns["underlying_price_col"]])
            else:
                intrinsic_value = 0  # Invalid option type, default to 0
            close_price = intrinsic_value
        else:
            # Use the regular close price
            close_price = self.get_close_price(row_close, position["trade_side"])

        # Calculate raw PnL
        if position["trade_side"] == "sell":
            # For short positions, profit = open price - close price
            raw_pnl = (position["pos_value"] - close_price) * position["quantity"] * 100
        else:
            # For long positions, profit = close price - open price
            raw_pnl = (close_price - position["pos_value"]) * position["quantity"] * 100

        # Log the "close" action with calculated PnL and intrinsic value
        self._log_daily_pnl(position=position, row=row_close, status="close", raw_pnl=raw_pnl, close_price=close_price, pnl_buffer=pnl_buffer)

        # Remove the position from open positions
        self.open_positions.remove(position)

    def close_all_positions(self, row_close, pnl_buffer):
        """
        Utility to close all open positions at once if desired.
        (Sometimes you might just want to close them simultaneously.)
        """
        # Copy because we'll modify the list inside the loop
        positions_copy = self.open_positions[:]
        for pos in positions_copy:
            self.close_position(pos, row_close, pnl_buffer)

    def _log_daily_pnl(self, position, row, status="hold", raw_pnl=None, close_price=None, pnl_buffer=None):
        """
        Create a daily log entry for a given position on a given day.

        - Opening fees: $0.75 per contract when a position is opened.
        - Closing fees: $0.75 per contract applied when a position is closed.
        - Fees are not compounded or multiplied unexpectedly.

        If pnl_buffer is provided, append the record to the buffer instead of writing immediately.
        """
        date_col = self.columns["date_col"]
        c_date = row[date_col].date()  # Use date object for dictionary keys

        # Determine the price for PnL logging
        if status == "open":
            pos_price = self.get_open_price(row, position["trade_side"])
            # Add opening fees based on quantity, but only once
            if not position.get("opening_fee_applied", False):
                opening_fee = self.opening_fee * position["quantity"]
                position["fees_incurred"] += opening_fee  # Fee per contract
                position["opening_fee_applied"] = True  # Mark fee as applied
        elif status == "close":
            pos_price = close_price if close_price is not None else self.get_close_price(row, position["trade_side"])
            # Add closing fees based on quantity, but only once
            if not position.get("closing_fee_applied", False):
                closing_fee = self.closing_fee * position["quantity"]
                position["fees_incurred"] += closing_fee  # Fee per contract
                position["closing_fee_applied"] = True  # Mark fee as applied
        else:  # "hold"
            pos_price = self.get_mid_price(row)
            # No additional fees during holding

        # Compute raw PnL if not provided
        if raw_pnl is None:
            if position["trade_side"] == "buy":
                # For long positions, PnL = (current price - open price) * quantity
                raw_pnl = (pos_price - position["pos_value"]) * position["quantity"] * 100
            else:
                # For short positions, PnL = (open price - current price) * quantity
                raw_pnl = (position["pos_value"] - pos_price) * position["quantity"] * 100

        # Adjust PnL by subtracting cumulative fees
        adjusted_pnl = raw_pnl - position["fees_incurred"]

        # Calculate percentage PnL
        cost_basis = position["pos_value"] * position["quantity"] * 100  # Total cost basis
        total_pct_pnl = adjusted_pnl / cost_basis if cost_basis != 0 else np.nan

        if pnl_buffer is not None:
            # Append the record to the buffer
            pnl_buffer.append({
                "date": row[date_col].strftime('%-m/%-d/%y'),
                "status": status,
                "symbol": position["symbol"],
                "side": "long" if position["trade_side"] == "buy" else "short",
                "call_put": position["call_put"],
                "expiration": position["expiration"].strftime('%-m/%-d/%y'),
                "strike": position["strike"],
                "delta": row[self.columns["delta_col"]],
                "underlying_price": row.get(self.columns["underlying_price_col"], np.nan),
                "pos_price": round(pos_price, 3),
                "quantity": position["quantity"],
                "openinterest": row.get(self.columns["openinterest_col"], np.nan),
                "iv": row.get(self.columns["iv_col"], np.nan),
                "total_raw_pnl": round(adjusted_pnl, 6),
                "total_pct_pnl": round(total_pct_pnl, 6),
                "fees_incurred": round(position["fees_incurred"], 6),
            })
        else:
            # Existing behavior: calculate combined totals and write to CSV
            # Update cumulative PnL and cost basis for the date
            self.date_pnl_summary[c_date] = self.date_pnl_summary.get(c_date, 0.0) + adjusted_pnl
            self.date_cost_basis[c_date] = self.date_cost_basis.get(c_date, 0.0) + cost_basis

            # Calculate combined PnL columns
            combined_total_raw_pnl = self.date_pnl_summary[c_date]
            combined_total_pct_pnl = (
                combined_total_raw_pnl / self.date_cost_basis[c_date] if self.date_cost_basis[c_date] != 0 else np.nan
            )

            # Create the log entry
            daily_record = {
                "date": row[date_col].strftime('%-m/%-d/%y'),
                "status": status,
                "symbol": position["symbol"],
                "side": "long" if position["trade_side"] == "buy" else "short",
                "call_put": position["call_put"],
                "expiration": position["expiration"].strftime('%-m/%-d/%y'),
                "strike": position["strike"],
                "delta": row[self.columns["delta_col"]],
                "underlying_price": row.get(self.columns["underlying_price_col"], np.nan),
                "pos_price": round(pos_price, 3),
                "quantity": position["quantity"],
                "openinterest": row.get(self.columns["openinterest_col"], np.nan),
                "iv": row.get(self.columns["iv_col"], np.nan),
                "total_raw_pnl": round(adjusted_pnl, 6),
                "total_pct_pnl": round(total_pct_pnl, 6),
                "fees_incurred": round(position["fees_incurred"], 6),
                "combined_total_raw_pnl": round(combined_total_raw_pnl, 6),
                "combined_total_pct_pnl": round(combined_total_pct_pnl, 6),
            }

            if self.daily_pnl_output_csv:
                with open(self.daily_pnl_output_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=daily_record.keys())
                    writer.writerow(daily_record)
            else:
                self.daily_pnl_log.append(daily_record)

    # ----------------------------------------------------------
    # Contract Selection
    # ----------------------------------------------------------
    
    def pick_contract(self, df_for_day, call_put, strike_selection="ATM"):
        """
        Select a contract based on the next expiration and strike selection criteria.
        
        :param df_for_day: DataFrame containing option data for the current day.
        :param call_put: 'C' for call options, 'P' for put options.
        :param strike_selection: "ATM" for at-the-money or a float for delta-based selection.
        :return: Series representing the selected option row or None if no suitable option is found.
        """
        # Filter by call/put
        sub = df_for_day[df_for_day[self.columns["call_put_col"]] == call_put]
        
        if sub.empty:
            logger.warning(f"No options found for {call_put} after filtering by call/put.")
            return None

        # Step 1: Filter by expiration (nearest expiration after current date)
        current_date = df_for_day[self.columns["date_col"]].iloc[0]
        sub = sub[sub[self.columns["expiration_col"]] > current_date]

        if self.monthly_expiration:
            # Apply monthly expiration filter (third Friday of the month)
            sub = sub[sub[self.columns["expiration_col"]].apply(self._is_third_friday)]

        if sub.empty:
            logger.warning(f"No options found for {call_put} with valid expirations.")
            return None

        # Sort by expiration to get the nearest expiration
        sub = sub.sort_values(by=self.columns["expiration_col"])
        nearest_expiration = sub[self.columns["expiration_col"]].iloc[0]

        # Filter for only the nearest expiration
        sub = sub[sub[self.columns["expiration_col"]] == nearest_expiration]

        # Step 2: Select strike based on strike selection criteria (ATM or delta)
        if strike_selection == "ATM":
            # Find the closest strike to the underlying price
            underlying_price = sub[self.columns["underlying_price_col"]].iloc[0]
            distance_atm = abs(sub[self.columns["strike_col"]] - underlying_price)
            best_option_idx = distance_atm.idxmin()
            best_option = sub.loc[best_option_idx]
        else:
            # Assume a float value is a delta and select the option with closest delta
            try:
                target_delta = float(strike_selection)
            except ValueError:
                raise TypeError(f"Invalid strike_selection value: {strike_selection}. Must be 'ATM' or a float.")

            # Compute distance to target delta for all options
            distance_delta = abs(sub[self.columns["delta_col"]] - target_delta)
            best_option_idx = distance_delta.idxmin()
            best_option = sub.loc[best_option_idx]

        logger.info(
            f"Selected {call_put} option: Expiration={best_option[self.columns['expiration_col']]}, "
            f"Strike={best_option[self.columns['strike_col']]}, Delta={best_option[self.columns['delta_col']]}"
        )
        return best_option

    # ----------------------------------------------------------
    # Main Backtest Loop
    # ----------------------------------------------------------
    
    def run_backtest(self):
        """Run the backtest, now capable of opening multiple positions at once."""
        csv_files = self._list_csv_files()
        for file_path in csv_files:
            df = self._read_data(file_path)
            unique_dates = df[self.columns["date_col"]].dt.date.unique()

            for i, d in enumerate(unique_dates):
                daily_df = df[df[self.columns["date_col"]].dt.date == d]
                current_date = pd.to_datetime(d)
                day_name = current_date.day_name()

                # Identify positions to close
                positions_to_close = []
                for pos in self.open_positions:
                    expiration_date = pos["expiration"].date()
                    day_before_expiration = expiration_date - pd.Timedelta(days=1)
                    if d >= expiration_date or (d == day_before_expiration and day_before_expiration.weekday() == 4):
                        positions_to_close.append(pos)

                # Buffer to hold all PnL records for the current date
                pnl_buffer = []

                # Close identified positions and log their PnL
                for pos in positions_to_close:
                    closing_rows = df[df[self.columns["date_col"]].dt.date <= pos["expiration"].date()]
                    if not closing_rows.empty:
                        row_close = closing_rows.iloc[-1]
                        self.close_position(pos, row_close, pnl_buffer)
                    else:
                        row_close = daily_df.iloc[-1]
                        self.close_position(pos, row_close, pnl_buffer)

                # Update daily PnL for open positions
                for pos in self.open_positions:
                    matching = daily_df[daily_df[self.columns["option_symbol_col"]] == pos["symbol"]]
                    if not matching.empty:
                        row = matching.iloc[0]
                        self._log_daily_pnl(position=pos, row=row, status="hold", pnl_buffer=pnl_buffer)

                # Open new positions if applicable
                if day_name == self.day_of_entry and len(self.open_positions) == 0:
                    self.open_multiple_positions(self.positions, daily_df, pnl_buffer)

                # Calculate combined PnL totals for the date
                if pnl_buffer:
                    # Calculate combined_total_raw_pnl
                    combined_total_raw_pnl = sum(record["total_raw_pnl"] for record in pnl_buffer)
                    # Calculate combined_total_pct_pnl
                    # To calculate percentage, we need the sum of cost basis for all positions on that date
                    combined_total_cost_basis = sum(
                        (record["pos_price"] * record["quantity"] * 100) for record in pnl_buffer
                    )
                    combined_total_pct_pnl = combined_total_raw_pnl / combined_total_cost_basis if combined_total_cost_basis != 0 else np.nan

                    # Update each record in the buffer with combined totals
                    for record in pnl_buffer:
                        record["combined_total_raw_pnl"] = round(combined_total_raw_pnl, 6)
                        record["combined_total_pct_pnl"] = round(combined_total_pct_pnl, 6)

                    # Write all buffered records to CSV
                    if self.daily_pnl_output_csv:
                        with open(self.daily_pnl_output_csv, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=[
                                "date", "status", "symbol", "side", "call_put", "expiration",
                                "strike", "delta", "underlying_price", "pos_price",
                                "quantity", "openinterest", "iv", "total_raw_pnl",
                                "total_pct_pnl", "fees_incurred",
                                "combined_total_raw_pnl", "combined_total_pct_pnl"
                            ])
                            writer.writerows(pnl_buffer)
                    else:
                        self.daily_pnl_log.extend(pnl_buffer)

                # Periodically collect garbage every 1000 iterations
                if i % 1000 == 0:
                    gc.collect()

        # No longer need to accumulate 'daily_pnl_log' in memory
        return

# ----------------------------------------------------------
# Example Usage
# ----------------------------------------------------------

# Define your positions with strike selection
positions = [
    {
        'side': 'buy',
        'callput': 'P',
        'quantity': 1,
        'strike_selection': -0.6  # 'ATM' or a float representing the desired delta (e.g., -0.6)
    },
    {
        'side': 'sell',
        'callput': 'P',
        'quantity': 2,
        'strike_selection': -0.4
    },
    {
        'side': 'sell',
        'callput': 'P',
        'quantity': 1,
        'strike_selection': -0.25
    },
]

if __name__ == "__main__":
    symbol = "SPY"
    backtester = RobustOptionBacktester(
        data_directory=f"historical/{symbol}/",
        positions=positions,
        start_date="2005-01-01",
        end_date="2024-12-31",
        day_of_entry="Monday",
        monthly_expiration=True,
        exclude_decimal_strikes=True,
        daily_pnl_output_csv=f"trade_logs/{symbol}_daily_pnl_log.csv",
        closed_positions_output_csv=f"trade_logs/{symbol}_closed_positions_log.csv",
    )
    backtester.run_backtest()