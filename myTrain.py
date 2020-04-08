from tqdm import tqdm
from utils.config import *
from models.model import *

# fixed random seed
if args['fixed']:
    torch.manual_seed(args['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        torch.backends.cudnn.deterministic = True
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])

# load data process function
early_stop = args['earlyStop']
if args['dataset'] == 'kvr':
    from utils.utils_Ent_kvr import *
    domains = {'navigate': 0, 'weather': 1, 'schedule': 2}
elif args['dataset'] == 'woz':
    from utils.utils_Ent_woz import *
    domains = {'restaurant': 0, 'attraction': 1, 'hotel': 2}
else:
    print("[ERROR] You need to provide the correct --dataset information")

# Configure models and load data
if args['epoch'] > 0:
    avg_best, cnt, res = 0.0, 0, 0.0
    train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))
    model = globals()['DFNet'](
        int(args['hidden']),
        lang,
        max_resp_len,
        args['path'],
        lr=float(args['learn']),
        n_layers=int(args['layer']),
        dropout=float(args['drop']),
        domains=domains)

    # Training
    for epoch in range(args['epoch']):
        print("Epoch:{}".format(epoch))
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data, int(args['clip']), reset=(i == 0))
            pbar.set_description(model.print_loss())
        if (epoch + 1) % int(args['evalp']) == 0:
            res = model.evaluate(dev, avg_best, early_stop=early_stop)
            model.scheduler.step(res)
            if res >= avg_best:
                avg_best = res
                cnt = 0
            else:
                cnt += 1
            if cnt == args['count']:
                print("Ran out of patient, early stop...")
                break

# Testing
train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))

model = globals()['DFNet'](
    int(args['hidden']),
    lang,
    max_resp_len,
    args['path'],
    lr=0.0,
    n_layers=int(args['layer']),
    dropout=0.0,
    domains=domains)

res_test = model.evaluate(test, 1e7, output=True)
