python log_processor.py "test_base.log" "test_300M.log" --cpu-load 0 --network-load 300 --temp-100 47 --temp-101 61
python simple_comparison.py --original_log "..\..\data\load_tests_0612\traffic_load\流量负载\log0-original_300M.log" --processed_log "test_300M.log" --output_dir "300M_cmp"




python log_processor.py "test_base.log" "test_cpu50.log" --cpu-load 50 --network-load 0 --temp-100 47 --temp-101 61
python simple_comparison.py --original_log "..\..\data\load_tests_0612\cpu_load\CPU负载\log0-original_cpu2.log" --processed_log "test_cpu50.log" --output_dir "cpu50_cmp"



python log_processor.py "test_base.log" "test_0M.log" --cpu-load 0 --network-load 0 --temp-100 47 --temp-101 61
python simple_comparison.py --original_log "test_base.log" --processed_log "test_0M.log" --output_dir "0M_cmp"

python log_processor.py "test_base.log" "test_200M.log" --cpu-load 0 --network-load 200 --temp-100 47 --temp-101 61
python simple_comparison.py --original_log "..\..\data\load_tests_0612\traffic_load\流量负载\log0-original_200M.log" --processed_log "test_200M.log" --output_dir "200M_cmp"
