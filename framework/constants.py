"""
Constants and features for DDoS Detection
"""

TIME_SPANS = {
    1: {
        'description': '1-second',
        'detailed_description': 'real-time resolution',
        'frequency': '1s',
        'pandas_floor': '1s'
    },
    10: {
        'description': '10-second',
        'detailed_description': 'ultra high-resolution',
        'frequency': '10s',
        'pandas_floor': '10s'
    },
    60: {
        'description': '1-minute', 
        'detailed_description': 'high-resolution',
        'frequency': '1min',
        'pandas_floor': 'min'
    },
    300: {
        'description': '5-minute',
        'detailed_description': 'standard resolution', 
        'frequency': '5min',
        'pandas_floor': '5min'
    }
}

SUPPORTED_TIME_SPANS = list(TIME_SPANS.keys())

FEATURE_GROUPS = {
    # Basic Features - Fundamental traffic metrics
    # total_flows: N (count of flows in time window)
    # total_packets: Σ packets_i for all flows i
    # total_bytes: Σ bytes_i for all flows i
    # avg_duration: (Σ duration_i) / N
    'basic': [
        'total_flows', 'total_packets', 'total_bytes', 'avg_duration'
    ],
    
    # Traffic Rates - Normalized traffic intensity metrics
    # packet_rate: total_packets / time_window (packets/second)
    # bit_rate: (total_bytes × 8) / time_window (bits/second)
    # flow_rate: total_flows / time_window (flows/second)
    # avg_packet_size: total_bytes / total_packets (bytes)
    # packets_per_flow: total_packets / total_flows
    # bytes_per_flow: total_bytes / total_flows
    'traffic_rates': [
        'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
        'packets_per_flow', 'bytes_per_flow'
    ],
    
    # Entropy Features - Shannon entropy H(X) = -Σ p(x) × log₂(p(x))
    # Measures randomness/diversity in distributions
    # src_port_entropy: H(source_ports) where p(port) = count(port) / total_flows
    # dst_port_entropy: H(destination_ports)
    # src_ip_entropy: H(source_ips)
    # src_as_entropy: H(source_AS_numbers)
    # src_geo_entropy: H(source_geolocation)
    # flow_size_entropy: H(flow_sizes) using binned byte counts
    # flag_combination_entropy: H(TCP_flag_combinations)
    'entropy': [
        'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy', 
        'src_as_entropy', 'src_geo_entropy', 'flow_size_entropy',
        'flag_combination_entropy'
    ],
    
    # Protocol Distribution - Protocol composition ratios
    # tcp_ratio: count(TCP_flows) / total_flows
    # udp_ratio: count(UDP_flows) / total_flows
    # icmp_ratio: count(ICMP_flows) / total_flows
    'protocol': [
        'tcp_ratio', 'udp_ratio', 'icmp_ratio'
    ],
    
    # IP Diversity - Source diversity metrics
    # unique_src_ips: |{source_ips}| (cardinality)
    # unique_src_as: |{source_AS_numbers}|
    # unique_src_geo: |{source_countries}|
    # as_diversity_ratio: unique_src_as / unique_src_ips
    # cross_border_ratio: count(src_country ≠ dst_country) / total_flows
    'ip_diversity': [
        'unique_src_ips', 'unique_src_as', 'unique_src_geo',
        'as_diversity_ratio', 'cross_border_ratio'
    ],
    
    # TCP Flags - TCP flag ratios
    # syn_flag_ratio: count(flows with SYN=1) / count(TCP_flows)
    # ack_flag_ratio: count(flows with ACK=1) / count(TCP_flows)
    # fin_flag_ratio: count(flows with FIN=1) / count(TCP_flows)
    # rst_flag_ratio: count(flows with RST=1) / count(TCP_flows)
    # psh_flag_ratio: count(flows with PSH=1) / count(TCP_flows)
    'tcp_flags': [
        'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 
        'rst_flag_ratio', 'psh_flag_ratio'
    ],
    
    # SYN Flood Specific - Features for detecting SYN flood attacks
    # syn_flood_ratio: count(SYN=1 and ACK=0) / count(TCP_flows)
    # syn_ack_ratio: count(SYN=1 and ACK=1) / count(TCP_flows)
    # small_packet_ratio: count(avg_packet_size < threshold) / total_flows
    # avg_tcp_duration: mean(duration for TCP_flows)
    # zero_duration_ratio: count(duration = 0) / total_flows
    # avg_ports_per_src: mean(|dst_ports per src_ip|)
    # max_ports_per_src: max(|dst_ports per src_ip|)
    'syn_flood_specific': [
        'syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio',
        'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
        'max_ports_per_src'
    ],
    
    # HTTP Flood Specific - Features for detecting HTTP flood attacks
    # large_packet_flow_ratio: count(avg_packet_size > threshold) / total_flows
    # size_uniformity: 1 - (std(bytes_per_flow) / mean(bytes_per_flow))
    # avg_dst_port_diversity: mean(H(dst_ports per src_ip))
    'http_flood_specific': [
        'large_packet_flow_ratio', 'size_uniformity', 'avg_dst_port_diversity'
    ],
    
    # Connection Patterns - TCP connection lifecycle metrics
    # connection_establishment_ratio: count(SYN=1) / count(TCP_flows)
    # connection_teardown_ratio: count(FIN=1 or RST=1) / count(TCP_flows)
    # incomplete_connection_ratio: count(SYN=1 and FIN=0 and RST=0) / count(TCP_flows)
    # zero_duration_connections: count(duration = 0)
    # very_short_connections: count(duration < 0.1s)
    'connection_patterns': [
        'connection_establishment_ratio', 'connection_teardown_ratio',
        'incomplete_connection_ratio', 'zero_duration_connections',
        'very_short_connections'
    ],
    
    # Inter-Arrival Times - Flow arrival timing patterns
    # Let Δt_i = timestamp_{i+1} - timestamp_i
    # avg_inter_arrival_time: mean(Δt)
    # std_inter_arrival_time: std(Δt)
    # min_inter_arrival_time: min(Δt)
    # max_inter_arrival_time: max(Δt)
    # inter_arrival_cv: std(Δt) / mean(Δt) (coefficient of variation)
    # burst_ratio: count(Δt < mean(Δt)/2) / N
    # periodic_pattern_score: based on autocorrelation of Δt
    # inter_arrival_variance: var(Δt)
    'inter_arrival': [
        'avg_inter_arrival_time', 'std_inter_arrival_time',
        'min_inter_arrival_time', 'max_inter_arrival_time',
        'inter_arrival_cv', 'burst_ratio', 'periodic_pattern_score',
        'inter_arrival_variance'
    ],
    
    # Flow Patterns - Flow size and packet distribution metrics
    # small_packet_flow_ratio: count(packets_per_flow < 5) / total_flows
    # single_packet_flow_ratio: count(packets_per_flow = 1) / total_flows
    # max_bytes_per_flow: max(bytes_i)
    # std_bytes_per_flow: std(bytes_i)
    # max_packets_per_flow: max(packets_i)
    # std_packets_per_flow: std(packets_i)
    'flow_patterns': [
        'small_packet_flow_ratio', 'single_packet_flow_ratio',
        'max_bytes_per_flow', 'std_bytes_per_flow',
        'max_packets_per_flow', 'std_packets_per_flow'
    ],
    
    # IP Behavior - Source IP behavioral patterns
    # high_volume_src_ratio: count(src_ips with flow_count > threshold) / unique_src_ips
    # max_flows_per_src: max(count(flows per src_ip))
    # avg_flows_per_src: total_flows / unique_src_ips
    # avg_src_port_diversity: mean(|src_ports per src_ip|)
    # port_scanning_ratio: count(src_ips contacting > N_threshold dst_ports) / unique_src_ips
    'ip_behavior': [
        'high_volume_src_ratio', 'max_flows_per_src', 'avg_flows_per_src',
        'avg_src_port_diversity', 'port_scanning_ratio'
    ],
    
    # Traffic Anomaly - Traffic rate variability metrics
    # packet_rate_std: std(packet_rate over time)
    # packet_rate_cv: std(packet_rate) / mean(packet_rate) (coefficient of variation)
    # flows_per_second: total_flows / time_window
    'traffic_anomaly': [
        'packet_rate_std', 'packet_rate_cv', 'flows_per_second'
    ],
    
    # Connection Diversity - Port diversity per IP
    # avg_src_ports_per_ip: mean(|src_ports| per src_ip)
    # avg_dst_ports_per_ip: mean(|dst_ports| per dst_ip)
    'connection_diversity': [
        'avg_src_ports_per_ip', 'avg_dst_ports_per_ip'
    ],
    
    # Statistical Features - Moment-based statistical measures
    # For any metric X (bytes, packets, duration):
    # X_mean: μ = (Σ x_i) / N
    # X_variance: σ² = (Σ(x_i - μ)²) / N
    # X_std: σ = √(σ²)
    # X_min: min(x_i)
    # X_max: max(x_i)
    # X_range: max(x_i) - min(x_i)
    # X_cv: σ / μ (coefficient of variation)
    # X_skewness: E[(X-μ)³] / σ³ (measures asymmetry)
    # X_kurtosis: E[(X-μ)⁴] / σ⁴ (measures tail heaviness)
    'statistical': [
        'bytes_mean', 'bytes_variance', 'bytes_std', 'bytes_min', 'bytes_max',
        'bytes_range', 'bytes_cv', 'bytes_skewness', 'bytes_kurtosis',
        'packets_mean', 'packets_variance', 'packets_std', 'packets_min', 'packets_max',
        'packets_range', 'packets_cv', 'packets_skewness', 'packets_kurtosis',
        'duration_mean', 'duration_variance', 'duration_std', 'duration_min', 'duration_max',
        'duration_range', 'duration_cv', 'duration_skewness', 'duration_kurtosis'
    ],
    
    # Diversity Indices - Statistical diversity measures
    # Simpson Index: D = 1 - Σ(n_i/N)² where n_i is count of category i
    #   Measures probability that two randomly selected items are different
    #   Applied to: src_ip, dst_ip, src_port, dst_port
    # Gini Index: G = (Σ Σ |x_i - x_j|) / (2N²μ)
    #   Measures inequality in distribution (0 = perfect equality, 1 = maximum inequality)
    #   Applied to: bytes, packets
    'diversity_indices': [
        'src_ip_simpson_index', 'dst_ip_simpson_index',
        'src_port_simpson_index', 'dst_port_simpson_index',
        'bytes_gini_index', 'packets_gini_index'
    ],
    
    # Uniqueness - Cardinality and uniqueness metrics
    # X_unique_count: |{unique values}| (cardinality)
    # X_uniqueness_ratio: unique_count / total_count
    # X_collision_probability: 1 - (unique_count / total_count)
    'uniqueness': [
        'src_ip_unique_count', 'src_ip_uniqueness_ratio', 'src_ip_collision_probability',
        'dst_ip_unique_count', 'dst_ip_uniqueness_ratio', 'dst_ip_collision_probability'
    ],
    
    # Fan-In/Fan-Out - Connection distribution patterns
    # max_fan_in: max(count(src_ips per dst_ip))
    # avg_fan_in: mean(count(src_ips per dst_ip))
    # fan_in_std: std(count(src_ips per dst_ip))
    # max_fan_out: max(count(dst_ips per src_ip))
    # avg_fan_out: mean(count(dst_ips per src_ip))
    # fan_out_std: std(count(dst_ips per src_ip))
    # fan_in_out_ratio: avg_fan_in / avg_fan_out
    'fan_in_out': [
        'max_fan_in', 'avg_fan_in', 'fan_in_std',
        'max_fan_out', 'avg_fan_out', 'fan_out_std',
        'fan_in_out_ratio'
    ],
    
    # Asymmetry - Entropy differences between source and destination
    # src_dst_port_entropy_diff: H(src_ports) - H(dst_ports)
    # src_dst_ip_entropy_diff: H(src_ips) - H(dst_ips)
    # port_entropy_abs_diff: |H(src_ports) - H(dst_ports)|
    # ip_entropy_abs_diff: |H(src_ips) - H(dst_ips)|
    'asymmetry': [
        'src_dst_port_entropy_diff', 'src_dst_ip_entropy_diff',
        'port_entropy_abs_diff', 'ip_entropy_abs_diff'
    ],
    
    # Amplification - DDoS amplification attack indicators
    # avg_request_reply_ratio: mean(response_bytes / request_bytes) for bidirectional flows
    # small_request_large_reply_ratio: count(response_bytes / request_bytes > 10) / total_flows
    # amplification_score: max(response_bytes) / min(request_bytes) per protocol
    # udp_amplification_potential: Based on dst_port (DNS=53, NTP=123, SSDP=1900, etc.)
    'amplification': [
        'avg_request_reply_ratio', 'small_request_large_reply_ratio',
        'amplification_score', 'udp_amplification_potential'
    ],
    
    # Concentration - Traffic destination concentration metrics
    # dst_top_N_ratio: sum(traffic to top N dst_ips) / total_traffic
    #   Measures how concentrated traffic is towards specific destinations
    # dst_heavy_tail_index: α parameter from Pareto distribution fit
    #   Higher α indicates less heavy-tailed distribution (α > 2 finite variance)
    # dst_concentration_score: Herfindahl-Hirschman Index HHI = Σ(s_i)²
    #   where s_i is market share of destination i (0 = perfect competition, 1 = monopoly)
    # Similar metrics for destination ports
    'concentration': [
        'dst_top_1_ratio', 'dst_top_5_ratio', 'dst_top_10_ratio',
        'dst_heavy_tail_index', 'dst_concentration_score',
        'dst_port_top_1_ratio', 'dst_port_top_5_ratio', 'dst_port_top_10_ratio',
        'dst_port_heavy_tail_index', 'dst_port_concentration_score'
    ],
    
    # Port Usage - Port range classification and concentration
    # ephemeral_port_ratio: count(port >= 49152) / total_flows
    # well_known_port_ratio: count(port <= 1023) / total_flows
    # registered_port_ratio: count(1024 <= port < 49152) / total_flows
    # dst_port_concentration: HHI for destination port distribution
    # src_port_concentration: HHI for source port distribution
    'port_usage': [
        'ephemeral_port_ratio', 'well_known_port_ratio', 'registered_port_ratio',
        'dst_port_concentration', 'src_port_concentration'
    ],
    
    # Entropy Difference - Entropy balance between source and destination
    # entropy_delta_src_dst_ip: H(src_ip) - H(dst_ip)
    # entropy_delta_src_dst_port: H(src_port) - H(dst_port)
    # entropy_balance_score: 1 - |entropy_delta| / max_possible_entropy
    #   Measures how balanced the entropy is between source and destination
    'entropy_diff': [
        'entropy_delta_src_dst_ip', 'entropy_delta_src_dst_port',
        'entropy_balance_score'
    ],
    
    # Spectral Features - Frequency domain analysis
    # spectral_entropy: H(FFT(traffic_signal))
    #   Entropy of the frequency spectrum, measures regularity in time series
    # traffic_energy: Σ|FFT(traffic)|²
    #   Total energy in frequency domain
    # traffic_burstiness: (σ² - μ) / (σ² + μ)
    #   Index of Dispersion, measures traffic burstiness (0 = periodic, >0 = bursty)
    'spectral': [
        'spectral_entropy', 'traffic_energy', 'traffic_burstiness'
    ],
    
    # Pure Diversity - Geographic and AS diversity metrics
    # as_diversity_count: |{unique AS numbers}|
    # as_diversity_ratio: as_diversity_count / total_flows
    # geo_diversity_count: |{unique countries}|
    # geo_diversity_ratio: geo_diversity_count / total_flows
    # cross_as_flow_ratio: count(src_AS ≠ dst_AS) / total_flows
    # cross_geo_flow_ratio: count(src_country ≠ dst_country) / total_flows
    'pure_diversity': [
        'as_diversity_count', 'as_diversity_ratio',
        'geo_diversity_count', 'geo_diversity_ratio',
        'cross_as_flow_ratio', 'cross_geo_flow_ratio'
    ],
    
    # EMA Smoothed Features - Exponentially weighted moving averages
    # EMA_t = α × value_t + (1-α) × EMA_{t-1}
    # where α = 2/(window+1) is the smoothing factor
    # Reduces noise and emphasizes recent trends in time-series features
    'ema_smoothed': [
        'packet_rate_ema',
        'bit_rate_ema',
        'flow_rate_ema',
        'duration_mean_ema',
        'bytes_std_ema',
        'packets_std_ema',
        'flows_per_second_ema'
    ]
}

FEATURES_BY_ATTACK_TYPE = {
    'all_features': {
        'include_groups': list(FEATURE_GROUPS.keys()),
        'exclude_features': []
    }
}
