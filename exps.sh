# Main results table
python main.py --use_react_chat --top_n 3 --seed 1
python main.py --use_react_chat --top_n 3 --seed 2
python main.py --use_react_chat --top_n 3 --seed 3
python main.py --use_react_chat --top_n 3 --seed 4
python main.py --use_react_chat --top_n 3 --seed 5
python main.py --top_n 3 --seed 1
python main.py --top_n 3 --seed 2
python main.py --top_n 3 --seed 3
python main.py --top_n 3 --seed 4
python main.py --top_n 3 --seed 5
python main.py --top_n 3 --seed 1 --sample="random"
python main.py --top_n 3 --seed 2 --sample="random"
python main.py --top_n 3 --seed 3 --sample="random"
python main.py --top_n 3 --seed 4 --sample="random"
python main.py --top_n 3 --seed 5 --sample="random"
# Appendix table (no fallback ablation)
python main.py --top_n 5 --seed 1
python main.py --top_n 5 --seed 2
python main.py --top_n 5 --seed 3
python main.py --top_n 5 --seed 4
python main.py --top_n 5 --seed 5
python main.py --top_n 5 --seed 1 --no_random_fallback
python main.py --top_n 5 --seed 2 --no_random_fallback
python main.py --top_n 5 --seed 3 --no_random_fallback
python main.py --top_n 5 --seed 4 --no_random_fallback
python main.py --top_n 5 --seed 5 --no_random_fallback
python main.py --top_n 3 --seed 1 --no_random_fallback
python main.py --top_n 3 --seed 2 --no_random_fallback
python main.py --top_n 3 --seed 3 --no_random_fallback
python main.py --top_n 3 --seed 4 --no_random_fallback
python main.py --top_n 3 --seed 5 --no_random_fallback