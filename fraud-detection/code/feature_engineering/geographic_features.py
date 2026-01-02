"""
Geographic Features for Fraud Detection
Detects impossible travel, unusual locations, and geographic deviations.
"""
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

class GeographicFeatures:
    """Geographic-based fraud detection features."""
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in km."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    def impossible_travel(self, df, lat_col='latitude', lon_col='longitude', 
                          time_col='transaction_time', user_col='user_id', max_speed_kmh=900):
        """Detect impossible travel (distance > max possible speed)."""
        df = df.copy().sort_values([user_col, time_col])
        df['impossible_travel'] = 0
        
        for user in df[user_col].unique():
            user_df = df[df[user_col] == user]
            if len(user_df) < 2:
                continue
            
            for i in range(1, len(user_df)):
                prev_row = user_df.iloc[i-1]
                curr_row = user_df.iloc[i]
                
                distance = self.haversine_distance(
                    prev_row[lat_col], prev_row[lon_col],
                    curr_row[lat_col], curr_row[lon_col]
                )
                time_diff_hours = (curr_row[time_col] - prev_row[time_col]).total_seconds() / 3600
                
                if time_diff_hours > 0 and distance / time_diff_hours > max_speed_kmh:
                    df.loc[curr_row.name, 'impossible_travel'] = 1
        
        return df
    
    def location_deviation(self, df, lat_col='latitude', lon_col='longitude', user_col='user_id'):
        """Detect transactions from unusual locations."""
        df = df.copy()
        
        user_centroids = df.groupby(user_col)[[lat_col, lon_col]].mean()
        df['home_distance_km'] = 0.0
        
        for user in df[user_col].unique():
            user_mask = df[user_col] == user
            home_lat, home_lon = user_centroids.loc[user]
            
            distances = []
            for idx, row in df[user_mask].iterrows():
                dist = self.haversine_distance(
                    row[lat_col], row[lon_col], home_lat, home_lon
                )
                distances.append(dist)
            
            df.loc[user_mask, 'home_distance_km'] = distances
        
        # Flag unusual distances (> 3 standard deviations)
        df['location_anomaly'] = 0
        for user in df[user_col].unique():
            user_mask = df[user_col] == user
            distances = df.loc[user_mask, 'home_distance_km']
            mean_dist = distances.mean()
            std_dist = distances.std()
            df.loc[user_mask & (df['home_distance_km'] > mean_dist + 3*std_dist), 'location_anomaly'] = 1
        
        return df

if __name__ == '__main__':
    print("Geographic Features Module")
