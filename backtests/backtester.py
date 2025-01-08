import os
import pandas as pd
import numpy as np
import datetime

class RobustOptionBacktester:
    """
    An extensive backtester that:
      - Iterates EOD option data from CSV(s) in a directory
      - Filters by user-specified day to enter, monthly expiry, delta, etc.
      - Opens/closes positions, calculates PnL including fees
      - Produces only a day-by-day PnL log (trade log references removed).
    """
    def __init__(
        self,
        data_directory,
        start_date=None,
        end_date=None,
        day_of_entry="Monday",
        call_or_put="C",
        trade_side="sell",
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
        self.call_or_put = call_or_put.upper()
        self.trade_side = trade_side.lower()
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
        self.open_position = None
        self.daily_pnl_log = []


    # --------------------- File Handling --------------------- #

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
        """Default check for monthly expiration: 3rd Friday of the month."""
        return (dt.weekday() == 4) and (15 <= dt.day <= 21)

    # --------------------- Entry/Exit Prices --------------------- #

    def get_open_price(self, row):
        """Determine the open price based on trade side."""
        if self.trade_side == "buy":
            return row[self.columns["ask_col"]]
        else:
            return row[self.columns["bid_col"]]

    def get_close_price(self, row):
        """Determine the close price based on trade side."""
        if self.open_position and self.open_position["trade_side"] == "buy":
            return row[self.columns["bid_col"]]
        else:
            return row[self.columns["ask_col"]]

    def get_mid_price(self, row):
        """Compute the midpoint price from bid/ask."""
        return 0.5 * (row[self.columns["ask_col"]] + row[self.columns["bid_col"]])

    # --------------------- Position Management --------------------- #

    def open_new_position(self, row):
        """Store the open position details."""
        open_price = self.get_open_price(row)
        pos_type = "ask" if self.trade_side == "buy" else "bid"

        self.open_position = {
            "symbol": row[self.columns["option_symbol_col"]],
            "open_datetime": row[self.columns["date_col"]],
            "expiration": row[self.columns["expiration_col"]],
            "call_put": row[self.columns["call_put_col"]],
            "strike": row[self.columns["strike_col"]],
            "openinterest": row.get(self.columns["openinterest_col"], np.nan),
            "iv": row.get(self.columns["iv_col"], np.nan),
            "trade_side": self.trade_side,
            "pos_type": pos_type,
            "pos_value": float(open_price),
            "quantity": self.contract_size,
        }

        self._log_daily_pnl(row, status="open")

    def close_position(self, row):
        """Close the open position and calculate PnL."""
        if not self.open_position:
            return

        close_price = self.get_close_price(row)
        raw_pnl = (
            (self.open_position["pos_value"] - close_price) * self.contract_size
            if self.trade_side == "sell"
            else (close_price - self.open_position["pos_value"]) * self.contract_size
        )

        total_fees_incurred = 1.50

        self._log_daily_pnl(row, status="close", fees_incurred=total_fees_incurred)
        self.open_position = None

    def _log_daily_pnl(self, row, status="hold", fees_incurred=0.0):
        """Log daily PnL for the position, including fees incurred and precise percentages."""
        if not self.open_position:
            return

        is_expiration_day = (
            row[self.columns["date_col"]].date() == self.open_position["expiration"].date()
        )

        if is_expiration_day:
            if self.open_position["call_put"] == "C":
                pos_price = max(
                    0, row[self.columns["underlying_price_col"]] - self.open_position["strike"]
                )
            elif self.open_position["call_put"] == "P":
                pos_price = max(
                    0, self.open_position["strike"] - row[self.columns["underlying_price_col"]]
                )
            pos_type = "intrinsic"
        elif status == "open":
            pos_price = self.get_open_price(row)
            pos_type = "ask" if self.trade_side == "buy" else "bid"
            fees_incurred = self.opening_fee
        elif status == "close":
            pos_price = self.get_close_price(row)
            pos_type = "mid"
            fees_incurred = self.opening_fee + self.closing_fee
        else:
            pos_price = self.get_mid_price(row)
            pos_type = "mid"
            fees_incurred = 0.0

        cumulative_fees = self.opening_fee
        if status == "close":
            cumulative_fees += self.closing_fee

        if self.trade_side == "buy":
            total_raw_pnl = (pos_price - self.open_position["pos_value"]) * 100
        else:
            total_raw_pnl = (self.open_position["pos_value"] - pos_price) * 100

        total_raw_pnl -= cumulative_fees
        original_pnl = self.open_position["pos_value"] * 100

        total_pct_pnl = (total_raw_pnl / original_pnl) if original_pnl != 0 else np.nan

        daily_record = {
            "date": row[self.columns["date_col"]],
            "status": status,
            "symbol": self.open_position["symbol"],
            "side": "short" if self.trade_side == "sell" else "long",
            "expiration": self.open_position["expiration"],
            "strike": self.open_position["strike"],
            "underlying_price": row.get(self.columns["underlying_price_col"], np.nan),
            "pos_type": pos_type,
            "pos_price": pos_price,
            "quantity": self.open_position["quantity"],
            "openinterest": row.get(self.columns["openinterest_col"], np.nan),
            "iv": row.get(self.columns["iv_col"], np.nan),
            "total_raw_pnl": round(total_raw_pnl, 8),
            "total_pct_pnl": round(total_pct_pnl, 4),
        }

        self.daily_pnl_log.append(daily_record)
        
    # --------------------- Contract Selection --------------------- #

    def pick_contract(self, df_for_day):
        """Pick the option contract based on nearest expiration and closest-to-ATM."""
        sub = df_for_day[df_for_day[self.columns["call_put_col"]] == self.call_or_put]

        if self.monthly_expiration:
            sub = sub[sub[self.columns["expiration_col"]].apply(self._is_third_friday)]
        if sub.empty:
            return None

        if self.exclude_decimal_strikes:
            sub = sub[sub[self.columns["strike_col"]] == sub[self.columns["strike_col"]].round()]
        if sub.empty:
            return None

        sub["days_to_expiration"] = (sub[self.columns["expiration_col"]] - sub[self.columns["date_col"]]).dt.days
        nearest_expiration = sub.loc[sub["days_to_expiration"].idxmin(), self.columns["expiration_col"]]

        sub = sub[sub[self.columns["expiration_col"]] == nearest_expiration]

        return sub.loc[sub[self.columns["otm_col"]].abs().idxmin()]

    # --------------------- Main Backtest Loop --------------------- #

    def run_backtest(self):
        """Run the backtesting process."""
        csv_files = self._list_csv_files()
        previous_expiration_date = None

        for file_path in csv_files:
            df = self._read_data(file_path)
            unique_dates = df[self.columns["date_col"]].dt.date.unique()

            for d in unique_dates:
                daily_df = df[df[self.columns["date_col"]].dt.date == d]
                current_date = pd.to_datetime(d)
                day_name = current_date.day_name()

                if previous_expiration_date is None:
                    third_fridays = daily_df[self.columns["expiration_col"]].apply(self._is_third_friday)
                    if third_fridays.any():
                        previous_expiration_date = daily_df.loc[third_fridays.idxmax(), self.columns["expiration_col"]].date()
                    continue

                monday_after_expiration = previous_expiration_date + pd.Timedelta(days=(7 - previous_expiration_date.weekday()))
                if current_date.date() < monday_after_expiration:
                    if self.open_position:
                        matching = daily_df[daily_df[self.columns["option_symbol_col"]] == self.open_position["symbol"]]
                        if not matching.empty:
                            self._log_daily_pnl(matching.iloc[0], status="hold")
                    continue

                if self.open_position and d >= self.open_position["expiration"].date():
                    previous_expiration_date = self.open_position["expiration"].date()
                    row_close = daily_df.iloc[-1]
                    self.close_position(row_close)
                    continue

                if self.open_position is None and day_name == self.day_of_entry:
                    candidate = self.pick_contract(daily_df)
                    if candidate is not None:
                        self.open_new_position(candidate)
                else:
                    if self.open_position:
                        matching = daily_df[daily_df[self.columns["option_symbol_col"]] == self.open_position["symbol"]]
                        if not matching.empty:
                            self._log_daily_pnl(matching.iloc[0], status="hold")

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
        start_date="2014-01-01",
        end_date="2024-12-31",
        call_or_put="P",
        trade_side="sell",
        monthly_expiration=True,
        exclude_decimal_strikes=True,
        daily_pnl_output_csv=f"trade_logs/{symbol}_daily_pnl_log.csv",
        closed_positions_output_csv=f"trade_logs/{symbol}_closed_positions_log.csv",
    )
    daily_df = backtester.run_backtest()