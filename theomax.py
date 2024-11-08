import pandas as pd
import numpy as np


def avg_battery_storage(data, battery_capacity):
    """
    Optimize battery storage and power sales based on price fluctuations.
    
    Parameters:
    data: pandas DataFrame with columns ['power_generated', 'price']
    battery_capacity: float, maximum storage capacity in MWh
    
    Returns:
    DataFrame with optimization results including battery level and revenue
    """
    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=['power_generated', 'price'])
    
    # Initialize results tracking
    results = []
    battery_level = 0
    total_revenue = 0
    
    # Calculate moving average price for decision making
    window_size = 24  # 24-hour window for price averaging
    data['price_ma'] = data['price'].rolling(window=window_size, center=True).mean()
    # Fill NaN values at edges with the overall mean
    data['price_ma'] = data['price_ma'].fillna(data['price'].mean())
    
    for idx, row in data.iterrows():
        power = row['power_generated']
        current_price = row['price']
        price_ma = row['price_ma']
        
        # Initialize hour's results
        power_sold = 0
        power_stored = 0
        
        # If price is above average, sell from battery
        if current_price > price_ma and battery_level > 0:
            # Sell all battery content
            power_sold = battery_level + power
            revenue = power_sold * current_price
            battery_level = 0
            
        # If price is below average and we have space in battery, store power
        elif current_price < price_ma:
            # Calculate how much we can store
            space_in_battery = battery_capacity - battery_level
            power_to_store = min(power, space_in_battery)
            
            # Store what we can
            power_stored = power_to_store
            battery_level += power_stored
            
            # Sell any excess that we couldn't store
            power_sold = power - power_stored
            revenue = power_sold * current_price
            
        # If price is average or we can't store, sell all power
        else:
            power_sold = power
            revenue = power_sold * current_price
        
        total_revenue += revenue
        
        # Store results for this hour
        results.append({
            'hour': idx,
            'power_generated': power,
            'price': current_price,
            'price_ma': price_ma,
            'power_stored': power_stored,
            'power_sold': power_sold,
            'battery_level': battery_level,
            'revenue': revenue,
            'total_revenue': total_revenue
        })
    
    return total_revenue.item()

def seventyfive_battery_storage(df, max_storage=50):
    """
    Optimize energy storage and sales for maximum profit
    """
    # Initialize storage and results
    current_storage = 0
    results = []

    # Convert data types and handle missing values

    # Calculate price threshold (75th percentile)
    price_threshold = df['price'].quantile(0.75)

    # Process each hour
    for index, row in df.iterrows():
        current_gen = row['power_generated']
        current_price = row['price']

        # Skip if we have missing data
        if pd.isna(current_gen) or pd.isna(current_price):
            continue

        # Strategy decision
        if current_storage < max_storage and current_price < price_threshold:
            # Store energy if price is low
            storage_capacity = min(current_gen, max_storage - current_storage)
            current_storage += storage_capacity
            energy_sold = current_gen - storage_capacity
        else:
            # Sell energy if price is high or storage is full
            energy_sold = current_gen + current_storage
            current_storage = 0

        # Record results
        results.append({
            'Hour': index + 1,
            'Original_Generation': current_gen,
            'Price': current_price,
            'Energy_Sold': energy_sold,
            'Storage_Level': current_storage,
            'Revenue': energy_sold * current_price
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate and print summary statistics
    original_revenue = sum(row['Original_Generation'] * row['Price'] for _, row in results_df.iterrows())
    optimized_revenue = results_df['Revenue'].sum()
    improvement = ((optimized_revenue - original_revenue) / original_revenue) * 100


    return optimized_revenue.item()

def theomax_battery_storage(data, battery_capacity):
    """
    Optimize battery storage and power selling decisions based on future prices.
    
    Args:
        power_generation (list): Hourly power generation in MWh
        prices (list): Hourly prices per MWh
        battery_capacity (float): Maximum battery capacity in MWh
    
    Returns:
        tuple: (total_revenue, battery_levels, power_sold_immediately)
    """
    power_generation = data["power_generated"]
    prices = data["price"]
    n_hours = len(power_generation)
    battery_level = 0.0
    total_revenue = 0.0
    battery_levels = []
    power_sold_immediately = []

    
    def get_future_window(current_hour):
        """Get the relevant future window for price comparison."""
        remaining_capacity = battery_capacity - battery_level
        # Calculate how many hours until battery would be full at max generation
        hours_to_full = float('inf')
        cumulative_power = 0
        
        for i in range(current_hour, min(current_hour + 100, n_hours)):
            cumulative_power += power_generation[i]
            if cumulative_power >= remaining_capacity:
                hours_to_full = i - current_hour + 1
                break
                
        # Window is minimum of 100 hours, hours until full, or remaining hours
        window_size = min(100, hours_to_full, n_hours - current_hour)
        return window_size
    
    for hour in range(n_hours):
        window_size = get_future_window(hour)
        future_prices = prices[hour:hour + window_size]
        
        current_power = power_generation[hour]
        current_price = prices[hour]
        
        # If no power generated this hour, skip
        if current_power == 0:
            battery_levels.append(battery_level)
            power_sold_immediately.append(0)
            continue
            
        # If current price is higher than all future prices in window, sell everything
        if current_price >= max(future_prices):
            power_to_sell = current_power + battery_level
            total_revenue += power_to_sell * current_price
            battery_level = 0
            battery_levels.append(battery_level)
            power_sold_immediately.append(power_to_sell)
            continue
            
        # If battery is full, must sell all current generation
        if battery_level >= battery_capacity:
            total_revenue += current_power * current_price
            battery_levels.append(battery_level)
            power_sold_immediately.append(current_power)
            continue
            
        # Calculate how much power can be stored
        space_in_battery = battery_capacity - battery_level
        power_to_store = min(current_power, space_in_battery)
        
        # Store power and sell any excess
        battery_level += power_to_store
        excess_power = current_power - power_to_store
        
        if excess_power > 0:
            total_revenue += excess_power * current_price
            
        battery_levels.append(battery_level)
        power_sold_immediately.append(excess_power)
    
    return total_revenue.item()

def demonstrate_optimization(data_path, battery_capacity):
    """
    Demonstrate the optimization algorithm with given data and battery capacity.
    """
    # Load data
    data = pd.read_csv(data_path, header=None, names=['power_generated', 'price'])

    # Run optimization
    results = (avg_battery_storage(data, battery_capacity), seventyfive_battery_storage(data, battery_capacity),theomax_battery_storage(data, battery_capacity))

    return results


def optimize(data,bound):
    c = 0
    p = 0
    for i in range(0, 1001):
        np = demonstrate_optimization(data, i)[bound] - 4000*i
        if(np > p):
            c = i
            p = np
#        print(c, p)

    print(data)
    print("Capacity: ", c)
    print("Total Revenue minus Cost: ", p)
    print("Total Revenue: ", p + 4000*c)

print("TheoMax")
optimize("valentino.csv", 2)
optimize("mantero.csv", 2)
optimize("howlinggale.csv", 2)
optimize("ventusvillage.csv", 2)
optimize("salmonvalley.csv", 2)

print("RollAvg")
optimize("valentino.csv", 0)
optimize("mantero.csv", 0)
optimize("howlinggale.csv", 0)
optimize("ventusvillage.csv", 0)
optimize("salmonvalley.csv", 0)

print("75Percentile")
optimize("valentino.csv", 1)
optimize("mantero.csv", 1)
optimize("howlinggale.csv", 1)
optimize("ventusvillage.csv", 1)
optimize("salmonvalley.csv", 1)
'''
demonstrate_optimization("data.csv",25)
demonstrate_optimization("data.csv",50)
demonstrate_optimization("data.csv",100)
demonstrate_optimization("data.csv",1000)
np = demonstrate_optimization("ventusvillage.csv",500)
print("Capacity: 500")
print("Total Revenue minus Cost: ", np[1] - 2000000)
print("Total Revenue: ", np[1])
print(np)
'''
