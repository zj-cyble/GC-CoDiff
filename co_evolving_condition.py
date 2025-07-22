import os
from absl import flags
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import tabular_dataload
# import tabular_dataloadPhishing1
import tabular_dataload1
from torch.utils.data import DataLoader
from models.tabular_unet import tabularUnet
from diffusion_discrete import MultinomialDiffusion
import evaluation
import logging
import numpy as np
import pandas as pd
from utils import *
import torch.nn as nn
from src.models.transformer_model import GraphTransformer
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport


def compute_shapes_trends(train, fake, metadata):
    qual_report = QualityReport()
    qual_report.generate(train, fake, metadata)
    shapes = qual_report.get_details(property_name='Column Shapes')
    trends = qual_report.get_details(property_name='Column Pair Trends')
    return shapes, trends


import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_alpha_precision(train, fake, test, k=5):
    nbrs_train = NearestNeighbors(n_neighbors=k + 1).fit(train)
    dist_train, _ = nbrs_train.kneighbors(train)
    dist_train = dist_train[:, 1:]  # 排除自身，取k个邻居
    dist_test, _ = nbrs_train.kneighbors(test)
    dist_fake, _ = nbrs_train.kneighbors(fake)
    thresh = np.percentile(dist_train, 95)  # train中95%邻居距离作为阈值
    alpha_precision = np.mean(np.percentile(dist_fake, 50, axis=1) < thresh)
    return alpha_precision


def compute_beta_recall(train, fake, test, k=5):
    nbrs_fake = NearestNeighbors(n_neighbors=k + 1).fit(fake)
    dist_fake_to_train, _ = nbrs_fake.kneighbors(train)
    dist_fake_to_train = dist_fake_to_train[:, 1:]  # 排除自身
    thresh = np.percentile(dist_fake_to_train, 95)
    beta_recall = np.mean(np.percentile(dist_fake_to_train, 50, axis=1) < thresh)
    return beta_recall


def train(FLAGS):
    FLAGS = flags.FLAGS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(FLAGS)
    train1, train_con_data1, train_dis_data1, test1, (transformer_con1, transformer_dis1, meta1), con_idx1, dis_idx1 = tabular_dataload1.get_dataset(FLAGS)
    train_iter_con = DataLoader(train_con_data, batch_size=FLAGS.training_batch_size)
    train_iter_dis = DataLoader(train_dis_data, batch_size=FLAGS.training_batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    train_iter_con1 = DataLoader(train_con_data1, batch_size=FLAGS.training_batch_size)
    train_iter_dis1 = DataLoader(train_dis_data1, batch_size=FLAGS.training_batch_size)
    datalooper_train_con1 = infiniteloop(train_iter_con1)
    datalooper_train_dis1 = infiniteloop(train_iter_dis1)
    num_class=[]
    for i in transformer_dis.output_info:
        num_class.append(i[0])
    num_class = np.array(num_class)
    if meta['problem_type'] == 'binary_classification':
        metric = 'binary_f1'
    elif meta['problem_type'] == 'regression': metric = "r2"
    else: metric = 'macro_f1'
    FLAGS.input_size = train_con_data.shape[1]
    FLAGS.cond_size = train_dis_data.shape[1]
    FLAGS.output_size = train_con_data.shape[1]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_con.split(',')))
    FLAGS.nf =  FLAGS.nf_con
    model_con = tabularUnet(FLAGS)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=FLAGS.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.mean_type, FLAGS.var_type).to(device)

    FLAGS.input_size = train_dis_data.shape[1]
    FLAGS.cond_size = train_con_data.shape[1]
    FLAGS.output_size = train_dis_data.shape[1]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_dis.split(',')))
    FLAGS.nf =  FLAGS.nf_dis
    model_dis = tabularUnet(FLAGS)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=FLAGS.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, FLAGS, timesteps=FLAGS.T,loss_type='vb_stochastic').to(device)

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)



    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    logging.info('Continuous model params: %d' % (num_params_con))
    logging.info('Discrete model params: %d' % (num_params_dis))

    scores_max_eval = -10

    total_steps_both = FLAGS.total_epochs_both * int(train.shape[0]/FLAGS.training_batch_size+1)
    sample_step = FLAGS.sample_step * int(train.shape[0]/FLAGS.training_batch_size+1)
    logging.info("Total steps: %d" %total_steps_both)
    logging.info("Sample steps: %d" %sample_step)
    logging.info("Continuous: %d, %d" %(train_con_data.shape[0], train_con_data.shape[1]))
    logging.info("Discrete: %d, %d"%(train_dis_data.shape[0], train_dis_data.shape[1]))

    # Start Training
    if FLAGS.eval==False:
        epoch = 0
        train_iter_con = DataLoader(train_con_data, batch_size=FLAGS.training_batch_size)
        train_iter_dis = DataLoader(train_dis_data, batch_size=FLAGS.training_batch_size)
        datalooper_train_con = infiniteloop(train_iter_con)
        datalooper_train_dis = infiniteloop(train_iter_dis)

        train_iter_con1 = DataLoader(train_con_data1, batch_size=FLAGS.training_batch_size)
        train_iter_dis1 = DataLoader(train_dis_data1, batch_size=FLAGS.training_batch_size)
        datalooper_train_con1 = infiniteloop(train_iter_con1)
        datalooper_train_dis1 = infiniteloop(train_iter_dis1)
        writer = SummaryWriter(FLAGS.logdir)
        writer.flush()

        for step in range(total_steps_both):
            model_con.train()
            model_dis.train()

            x_0_con = next(datalooper_train_con).to(device).float()
            x_0_dis = next(datalooper_train_dis).to(device)
            ns_con = next(datalooper_train_con1).to(device).float()
            ns_dis = next(datalooper_train_dis1).to(device)
            con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, transformer_dis, FLAGS)

            loss_con = con_loss + FLAGS.lambda_con * con_loss_ns
            loss_dis = dis_loss + FLAGS.lambda_dis * dis_loss_ns

            optim_con.zero_grad()
            loss_con.backward()
            torch.nn.utils.clip_grad_norm_(model_con.parameters(), FLAGS.grad_clip)
            optim_con.step()
            sched_con.step()

            optim_dis.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), FLAGS.grad_clip)#, self.args.clip_value)
            torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), FLAGS.grad_clip)#, self.args.clip_norm)
            optim_dis.step()
            sched_dis.step()

            # log
            writer.add_scalar('loss_continuous', con_loss, step)
            writer.add_scalar('loss_discrete', dis_loss, step)
            writer.add_scalar('loss_continuous_ns', con_loss_ns, step)
            writer.add_scalar('loss_discrete_ns', dis_loss_ns, step)
            writer.add_scalar('total_continuous', loss_con, step)
            writer.add_scalar('total_discrete', loss_dis, step)

            if (step+1) % int(train.shape[0]/FLAGS.training_batch_size+1) == 0:

                logging.info(f"Epoch :{epoch}, diffusion continuous loss: {con_loss:.3f}, discrete loss: {dis_loss:.3f}")
                logging.info(f"Epoch :{epoch}, CL continuous loss: {con_loss_ns:.3f}, discrete loss: {dis_loss_ns:.3f}")
                logging.info(f"Epoch :{epoch}, Total continuous loss: {loss_con:.3f}, discrete loss: {loss_dis:.3f}")
                epoch +=1

            if step > 0 and sample_step > 0 and step % sample_step == 0 or step==(total_steps_both-1):
                model_con.eval()
                model_dis.eval()
                with torch.no_grad():
                    x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
                    log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
                    x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
                x_dis = apply_activate(x_dis, transformer_dis.output_info)
                sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
                sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
                sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])
                for i in range(len(con_idx)):
                    sample[:,con_idx[i]]=sample_con[:,i]
                for i in range(len(dis_idx)):
                    sample[:,dis_idx[i]]=sample_dis[:,i]
                sample = np.array(pd.DataFrame(sample).dropna())
                scores, std, param = evaluation.compute_scores(train=train, test = None, synthesized_data=[sample], metadata=meta, eval=None)
                div_mean, div_std = evaluation.compute_diversity(train=train, fake=[sample])
                f1 = scores[metric]
                logging.info(f"---------Epoch {epoch} Evaluation----------")
                logging.info(scores)
                logging.info(std)

                if scores_max_eval < torch.tensor(f1):
                    scores_max_eval = torch.tensor(f1)
                    logging.info(f"Save model!")
                    ckpt = {
                        'model_con': model_con.state_dict(),
                        'model_dis': model_dis.state_dict(),
                        'sched_con': sched_con.state_dict(),
                        'sched_dis': sched_dis.state_dict(),
                        'optim_con': optim_con.state_dict(),
                        'optim_dis': optim_dis.state_dict(),
                        'step': step,
                        'sample': sample, 
                        'ml_param': param
                    }
                    torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
        logging.info(f"Evaluation best : {scores_max_eval}")

        #final test
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        model_con.load_state_dict(ckpt['model_con'])
        model_dis.load_state_dict(ckpt['model_dis'])
        model_con.eval()
        model_dis.eval()
        fake_sample=[]
        for i in range(5):
            logging.info(f"sampling {i}")
            with torch.no_grad():
                x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
                log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
                x_con, x_dis= sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
            x_dis = apply_activate(x_dis, transformer_dis.output_info)
            sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
            sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
            sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])
            for i in range(len(con_idx)):
                sample[:,con_idx[i]]=sample_con[:,i]
            for i in range(len(dis_idx)):
                sample[:,dis_idx[i]]=sample_dis[:,i]
            fake_sample.append(sample)
        for i, f in enumerate(fake_sample):
            df = pd.DataFrame(f)
            df.to_csv(f"/home/generated_samples/bei0030xin_{i}.csv",
                      index=False)
        scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        logging.info(f"---------Test----------")
        logging.info(scores)
        logging.info(std)

    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        model_con.load_state_dict(ckpt['model_con'])
        model_dis.load_state_dict(ckpt['model_dis'])
        model_con.eval()
        model_dis.eval()
        fake_sample = []
        for i in range(5):
            logging.info(f"sampling {i}")
            with torch.no_grad():
                x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
                log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
                x_con, x_dis= sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
            x_dis = apply_activate(x_dis, transformer_dis.output_info)
            sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
            sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
            sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])
            for i in range(len(con_idx)):
                sample[:,con_idx[i]]=sample_con[:,i]
            for i in range(len(dis_idx)):
                sample[:,dis_idx[i]]=sample_dis[:,i]
            fake_sample.append(sample)
        scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        logging.info(f"---------Test----------")
        logging.info(scores)
        logging.info(std)