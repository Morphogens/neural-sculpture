tmux new -d -s jupyter;
tmux send-keys -t jupyter "conda activate neural-sculpture; jupyter lab" ENTER;

tmux new -d -s http;
tmux send-keys -t http "cd experiments; python -m http.server 8001" ENTER;


tmux new -d -s app;
tmux send-keys -t app "cd client; npm run dev" ENTER;

tmux new -d -s server;
tmux send-keys -t server "conda activate neural-sculpture; python main_webserver.py" ENTER;

