import pyshark
import socket
import numpy as np
import threading


def read_pcap(file_path):
    #cap = pyshark.FileCapture(file_path)
    cap = pyshark.FileCapture(file_path , tshark_path='E:\\Wireshark\\tshark.exe')

    packets = []
    for packet in cap:
        try:
            src = packet.ip.src
            dst = packet.ip.dst
            proto = packet.transport_layer
            length = int(packet.length)

            print(f"Source: {src}, Destination: {dst}, Protocol: {proto}, Length: {length}")

            
            if proto == "TCP" and length < 100:
                print(f"[!] Possible botnet activity detected from {src} to {dst}")

            packets.append((src, dst, float(packet.sniff_time.timestamp())))  

        except AttributeError:
            continue  # Skip packets without IP or transport layers

    return packets

def is_cloud_server(ip):
    try:
        domain = socket.gethostbyaddr(ip)[0]
        cloud_keywords = ["amazonaws", "google", "microsoft", "azure", "cloud"]
        return any(keyword in domain for keyword in cloud_keywords)
    except:
        return False

# Filter bot-CnC communication
def filter_suspicious_ips(packets):
    suspicious_packets = []
    for src, dst, timestamp in packets:
        if not is_cloud_server(dst):
            suspicious_packets.append((src, dst, timestamp))
    return suspicious_packets

# Compute Autocorrelation Function (ACF) to detect periodicity in communication
def autocorrelation(timestamps, max_lag=10):
    diffs = np.diff(sorted(timestamps))
    if len(diffs) < max_lag:
        return False  # Not enough data points for analysis

    acf_values = [np.corrcoef(diffs[:-lag], diffs[lag:])[0, 1] for lag in range(1, max_lag)]
    return max(acf_values) > 0.7  # Threshold for periodicity detection

#(Parallel Threads)
def detect_bot_cnc_parallel(devices):
    infected_devices = []
    mid = len(devices) // 2

    def process_half(device_list):
        for device, timestamps in device_list.items():
            if autocorrelation(timestamps):
                infected_devices.append(device)

  
    thread1 = threading.Thread(target=process_half, args=(dict(list(devices.items())[:mid]),))
    thread2 = threading.Thread(target=process_half, args=(dict(list(devices.items())[mid:]),))

    
    thread1.start()
    thread2.start()

    
    thread1.join()
    thread2.join()

    return infected_devices


def main():
    pcap_file = "botnet_traffic.pcap"  # Change to your actual file path

    print("Reading packets from PCAP file...")
    packets = read_pcap(pcap_file)

    print("Filtering potential bot-CnC communication...")
    suspicious_packets = filter_suspicious_ips(packets)

    print("Analyzing periodicity to detect bots (Parallel Processing)...")
    
   
    device_traffic = {}
    for src, dst, timestamp in suspicious_packets:
        if src not in device_traffic:
            device_traffic[src] = []
        device_traffic[src].append(timestamp)

    infected_devices = detect_bot_cnc_parallel(device_traffic)

    if infected_devices:
        print("Detected infected IoT devices:", infected_devices)
    else:
        print("No infected devices detected.")

if __name__ == "__main__":
    main()
