# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8-char-6e-4lr-2e3warmup-debug-resume-resume-resume'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
init_from = "resume"
#  init_from = "scratch"

wandb_log = False # override via command line if you like
wandb_project = 'enwik8-205chars'
wandb_run_name = 'gpt2-124M-6e-4lr-25000warmup'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 1024 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 600000
lr_decay_iters = 600000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 25000 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
