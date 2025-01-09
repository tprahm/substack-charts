import os
import pandas as pd
import numpy as np
import datetime

class RobustOptionBacktester:
    """
    An extensive backtester that:
      - Iterates EOD option data from CSV(s) in a directory
      - Can open multiple positions in one day (e.g. sell put + sell call)
      - Calculates day-by-day PnL for each open position
      - Produces a day-by-day PnL log where each row can represent one position
        on a given date (i.e. potentially multiple rows per date).
    """
    def __init__(
        self,
        data_directory,
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
        
        # Instead of a single open_position, we maintain a list of open positions
        self.open_positions = []
        
        # Our daily log can now have multiple rows for a single date (one row per position)
        self.daily_pnl_log = []

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
    
    def open_new_position(self, row, trade_side):
        """
        Create a single new open-position dict for tracking.
        e.g. if user wants to short a Put, trade_side="sell", call_put="P".
        """
        open_price = self.get_open_price(row, trade_side)

        new_pos = {
            "symbol": row[self.columns["option_symbol_col"]],
            "open_datetime": row[self.columns["date_col"]],
            "expiration": row[self.columns["expiration_col"]],
            "call_put": row[self.columns["call_put_col"]],  # 'C' or 'P'
            "strike": row[self.columns["strike_col"]],
            "openinterest": row.get(self.columns["openinterest_col"], np.nan),
            "iv": row.get(self.columns["iv_col"], np.nan),
            "trade_side": trade_side.lower(),              # 'buy' or 'sell'
            "pos_value": float(open_price),
            "quantity": self.contract_size,
            "fees_incurred": self.opening_fee,             # Track opening fees
            "underlying_price_at_expiration": None,         # Placeholder for expiration price
        }

        self.open_positions.append(new_pos)
        
        # Log an 'open' record to daily PnL
        self._log_daily_pnl(position=new_pos, row=row, status="open")

    def open_multiple_positions(self, rows_and_sides):
        """
        If you want to open multiple legs simultaneously (e.g. short call + short put),
        you can pass a list of (row, trade_side) pairs here.
        
        Example usage:
          self.open_multiple_positions([
              (row_for_call, "sell"),
              (row_for_put,  "sell")
          ])
        """
        for (row, side) in rows_and_sides:
            self.open_new_position(row, side)

    def close_position(self, position, row_close):
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
            raw_pnl = (position["pos_value"] - close_price) * self.contract_size * 100
        else:
            # For long positions, profit = close price - open price
            raw_pnl = (close_price - position["pos_value"]) * self.contract_size * 100

        # Subtract fees (add closing fee to previously incurred opening fee)
        position["fees_incurred"] += self.closing_fee
        raw_pnl -= position["fees_incurred"]

        # Log the "close" action with calculated PnL and intrinsic value
        self._log_daily_pnl(position=position, row=row_close, status="close", raw_pnl=raw_pnl, close_price=close_price)

        # Remove the position from open positions
        self.open_positions.remove(position)

    def close_all_positions(self, row):
        """
        Utility to close all open positions at once if desired.
        (Sometimes you might just want to close them simultaneously.)
        """
        # Copy because we'll modify the list inside the loop
        positions_copy = self.open_positions[:]
        for pos in positions_copy:
            self.close_position(pos, row)

    def _log_daily_pnl(self, position, row, status="hold", raw_pnl=None, close_price=None):
        """
        Create a daily log entry for a given position on a given day.

        If status == "open", apply opening fees.
        If status == "close", do not apply additional fees here as they are handled in close_position.
        If status == "hold", apply holding fees if any (currently no additional fees).
        """
        date_col = self.columns["date_col"]
        c_date = row[date_col]
        
        # Determine the price for PnL logging
        if status == "open":
            pos_price = self.get_open_price(row, position["trade_side"])
            # Fees are already applied in open_new_position
        elif status == "close":
            pos_price = close_price if close_price is not None else self.get_close_price(row, position["trade_side"])
            # Fees are already applied in close_position
        else:  # "hold"
            pos_price = self.get_mid_price(row)
            # Fees remain unchanged during holding

        # Compute raw PnL if not provided
        if raw_pnl is None:
            if position["trade_side"] == "buy":
                raw_pnl = (pos_price - position["pos_value"]) * self.contract_size * 100
            else:
                raw_pnl = (position["pos_value"] - pos_price) * self.contract_size * 100

        # Subtract cumulative fees from raw PnL only if status is not "close"
        if status != "close":
            raw_pnl -= position["fees_incurred"]

        # Percentage PnL
        cost_basis = position["pos_value"] * 100
        total_pct_pnl = raw_pnl / cost_basis if cost_basis != 0 else np.nan

        # Create the log entry
        daily_record = {
            "date": c_date.strftime('%-m/%-d/%y'),
            "status": status,
            "symbol": position["symbol"],
            "side": "long" if position["trade_side"] == "buy" else "short",
            "call_put": position["call_put"],
            "expiration": position["expiration"].strftime('%-m/%-d/%y'),
            "strike": position["strike"],
            "underlying_price": row.get(self.columns["underlying_price_col"], np.nan),
            "pos_price": round(pos_price, 3),
            "quantity": position["quantity"],
            "openinterest": row.get(self.columns["openinterest_col"], np.nan),
            "iv": row.get(self.columns["iv_col"], np.nan),
            "total_raw_pnl": round(raw_pnl, 6),
            "total_pct_pnl": round(total_pct_pnl, 6),
        }

        self.daily_pnl_log.append(daily_record)

    # ----------------------------------------------------------
    # Contract Selection (example placeholders)
    # ----------------------------------------------------------
    
    def pick_contract(self, df_for_day, call_put):
        # Filter by call/put
        sub = df_for_day[df_for_day[self.columns["call_put_col"]] == call_put]
        
        # Print or log what you have
        print("---- Full sub for call_put=", call_put, "----")
        print(sub[[self.columns["expiration_col"], self.columns["strike_col"]]].drop_duplicates())

        if self.monthly_expiration:
            sub = sub[sub[self.columns["expiration_col"]].apply(self._is_third_friday)]
            print("---- After monthly filter ----")
            print(sub[[self.columns["expiration_col"], self.columns["strike_col"]]].drop_duplicates())

        # Filter only expirations after the current day
        current_date = df_for_day[self.columns["date_col"]].iloc[0]
        sub = sub[sub[self.columns["expiration_col"]] > current_date]
        print("---- After expiration > current_date filter ----")
        print(sub[[self.columns["expiration_col"], self.columns["strike_col"]]].drop_duplicates())

        if sub.empty:
            return None

        # Sort and pick the earliest expiration
        sub = sub.sort_values(by=self.columns["expiration_col"], ascending=True)
        nearest_expiration = sub.iloc[0][self.columns["expiration_col"]]
        sub = sub[sub[self.columns["expiration_col"]] == nearest_expiration]
        
        # Now pick closest to ATM
        sub["distance_atm"] = abs(sub[self.columns["strike_col"]] - sub[self.columns["underlying_price_col"]])
        row = sub.loc[sub["distance_atm"].idxmin()]
        
        return row

    # ----------------------------------------------------------
    # Main Backtest Loop (Corrected Logic)
    # ----------------------------------------------------------
    
    def run_backtest(self):
        """Run the backtest, now capable of opening multiple positions at once."""
        csv_files = self._list_csv_files()
        for file_path in csv_files:
            df = self._read_data(file_path)
            unique_dates = df[self.columns["date_col"]].dt.date.unique()

            for d in unique_dates:
                daily_df = df[df[self.columns["date_col"]].dt.date == d]
                current_date = pd.to_datetime(d)
                day_name = current_date.day_name()

                # 1) Identify positions to close (expiration date <= current date or adjusted expiration)
                positions_to_close = []
                for pos in self.open_positions:
                    expiration_date = pos["expiration"].date()
                    day_before_expiration = expiration_date - pd.Timedelta(days=1)
                    
                    # Check if the current date is expiration day or the adjusted expiration (Friday before)
                    if d >= expiration_date or (d == day_before_expiration and day_before_expiration.weekday() == 4):
                        positions_to_close.append(pos)

                # 2) Close identified positions using the last available underlying price before or on expiration
                for pos in positions_to_close:
                    # Find the last row on or before expiration date
                    closing_rows = df[df[self.columns["date_col"]].dt.date <= pos["expiration"].date()]
                    if not closing_rows.empty:
                        row_close = closing_rows.iloc[-1]
                        self.close_position(pos, row_close)
                    else:
                        # If no data on or before expiration, use current day's underlying price
                        row_close = daily_df.iloc[-1]
                        self.close_position(pos, row_close)

                # 3) Update daily PnL for all remaining open positions as "hold"
                for pos in self.open_positions:
                    matching = daily_df[daily_df[self.columns["option_symbol_col"]] == pos["symbol"]]
                    if not matching.empty:
                        row = matching.iloc[0]
                        self._log_daily_pnl(position=pos, row=row, status="hold")

                # 4) If it's the day_of_entry and no open positions, open new positions
                if day_name == self.day_of_entry and len(self.open_positions) == 0:
                    # For instance, open a 2-leg strangle: short call + short put.
                    row_call = self.pick_contract(daily_df, call_put="C")
                    row_put  = self.pick_contract(daily_df, call_put="P")
                    if (row_call is not None) and (row_put is not None):
                        # Sell the call and sell the put simultaneously
                        self.open_multiple_positions([
                            (row_call, "sell"), 
                            (row_put,  "sell")
                        ])

        # Convert daily log to DataFrame
        final_daily_df = pd.DataFrame(self.daily_pnl_log)

        # Save the daily PnL log
        if self.daily_pnl_output_csv:
            try:
                final_daily_df.to_csv(self.daily_pnl_output_csv, index=False)
                print(f"Daily PnL log saved successfully to: {self.daily_pnl_output_csv}")
            except Exception as e:
                print(f"Error saving daily PnL log to CSV: {e}")

        # Filter and save closed positions log
        if self.closed_positions_output_csv:
            closed_positions = final_daily_df[final_daily_df["status"] == "close"]
            try:
                closed_positions.to_csv(self.closed_positions_output_csv, index=False)
                print(f"Closed positions log saved successfully to: {self.closed_positions_output_csv}")
            except Exception as e:
                print(f"Error saving closed positions log to CSV: {e}")

        return final_daily_df

# --------------------- Example Usage --------------------- #
if __name__ == "__main__":
    symbol = "UVXY"
    backtester = RobustOptionBacktester(
        data_directory=f"historical/{symbol}/",
        start_date="2011-01-01",
        end_date="2018-12-31",
        day_of_entry="Monday",
        monthly_expiration=True,
        exclude_decimal_strikes=True,
        daily_pnl_output_csv=f"trade_logs/{symbol}_daily_pnl_log.csv",
        closed_positions_output_csv=f"trade_logs/{symbol}_closed_positions_log.csv",
    )
    daily_df = backtester.run_backtest()
    print(daily_df)