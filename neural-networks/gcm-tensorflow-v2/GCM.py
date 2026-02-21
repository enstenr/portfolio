import hashlib
import logging
import os
import random
import sys
from collections import OrderedDict
from time import localtime, strftime, time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers


from utility import DataGenerator, Learner, Tool, configs
from utility.AbstractRecommender import AbstractRecommender
from utility.DataIterator import DataIterator
from utility.Dataset import Dataset as DATA
from utility.Tool import ensureDir, timer

np.random.seed(2019)
random.seed(2019)
tf.random.set_seed(2019)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

class GCM(AbstractRecommender, tf.keras.Model):
    def __init__(self, dataset, conf):
        AbstractRecommender.__init__(self, dataset, conf)
        tf.keras.Model.__init__(self)

        self.dataset_name = dataset.dataset_name
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_valid_items = dataset.num_valid_items
        self.num_user_features = dataset.num_user_features
        self.num_item_features = dataset.num_item_featuers
        self.num_context_features = dataset.num_context_features
        self.dataset = dataset

        self.user_feature_mat = self.get_feature_matrix('user')
        self.item_feature_mat = self.get_feature_matrix('item')
        self.context_feature_mat = self.get_feature_matrix('context')
        self.num_contexts = len(self.context_feature_mat)

        self.context_feature_wo_last = self.context_feature_mat[:, :-1]
        self.context_feature_last = self.context_feature_mat[:, -1]

        self.insts2userid = dataset.all_data_dict['train_data']['user_id'].to_list()
        self.insts2itemid = dataset.all_data_dict['train_data']['item_id'].to_list()
        self.insts2contextid = dataset.all_data_dict['train_data']['context_id'].to_list()
        self.num_context_fields = self.dataset.side_info['side_info_stats']['num_context_fields']

        self.adj_norm_type = conf.adj_norm_type
        if self.adj_norm_type in ['rs', 'rd', 'db']:
            self.user_neighbor_num, self.item_neighbor_num = self.cnt_neighbour_number(
                dataset.all_data_dict['train_data'])
            self.norm_user_neighbor_num = self.get_inv_neighbor_num(self.user_neighbor_num, self.adj_norm_type)
            self.norm_item_neighbor_num = self.get_inv_neighbor_num(self.item_neighbor_num, self.adj_norm_type)

        self.batch_size = conf.batch_size
        self.test_batch_size = conf.test_batch_size
        self.learning_rate = conf.lr
        self.hidden_factor = conf.hidden_factor
        self.num_epochs = conf.epoch
        self.optimizer_type = conf.optimizer
        self.reg = conf.reg
        self.loss_type = conf.loss_type
        self.num_negatives = conf.num_negatives if self.loss_type == 'log_loss' else 0
        self.num_gcn_layers = conf.num_gcn_layers
        self.gcn_layer_weight = conf.gcn_layer_weight
        self.merge_type = conf.merge_type
        self.decoder_type = conf.decoder_type
        self.num_hidden_layers = conf.num_hidden_layers if self.decoder_type == 'MLP' else 0
        self.save_flag = conf.save_flag
        self.save_file = conf.save_file if hasattr(conf, 'save_file') else None

        self.best_result = np.zeros([9], dtype=float)
        self.best_epoch = 0

    def get_feature_matrix(self, key_word):
        mat = self.dataset.side_info['%s_feature' % key_word]
        return mat.values if mat is not None else None

    def cnt_neighbour_number(self, df):
        user_neighbor_num = np.zeros([self.num_users], dtype=int)
        item_neighbor_num = np.zeros([self.num_valid_items], dtype=int)
        for id, value in df['user_id'].value_counts().items(): user_neighbor_num[id] = value
        for id, value in df['item_id'].value_counts().items(): item_neighbor_num[id] = value
        return user_neighbor_num, item_neighbor_num

    def get_inv_neighbor_num(self, data, norm_type):
        p = -0.5 if norm_type in ['rs', 'db'] else -1.0
        d_inv = np.power(data, p).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        return d_inv

    def _create_variables(self):
        initializer = tf.keras.initializers.GlorotUniform()
        zero_init = tf.keras.initializers.Zeros()

        self.user_embeddings = self.add_weight(name='user_embed', shape=(self.num_users, self.hidden_factor),
                                               initializer=initializer)
        self.item_embeddings = self.add_weight(name='item_embed', shape=(self.num_items, self.hidden_factor),
                                               initializer=initializer)
        self.user_feature_embeddings = self.add_weight(name='u_feat_embed',
                                                       shape=(self.num_user_features, self.hidden_factor),
                                                       initializer=initializer)
        self.item_feature_embeddings = self.add_weight(name='i_feat_embed',
                                                       shape=(self.num_item_features, self.hidden_factor),
                                                       initializer=initializer)
        self.context_feature_embeddings = self.add_weight(name='c_feat_embed',
                                                          shape=(self.num_context_features - self.num_items,
                                                                 self.hidden_factor), initializer=initializer)

        self.user_bias = self.add_weight(name='u_bias', shape=(self.num_users, 1), initializer=zero_init)
        self.item_bias = self.add_weight(name='i_bias', shape=(self.num_items, 1), initializer=zero_init)
        self.global_bias = self.add_weight(name='g_bias', shape=(1, 1), initializer=zero_init)

    def call(self, inputs, training=None):
        user_indices = inputs['user_id']
        item_indices = inputs['item_id']
        context_indices = inputs['context_id']

        # Flatten context_id to ensure we don't get an extra dimension during gather
        # Force context indices to 1D
        context_indices = tf.reshape(inputs['context_id'], [-1])
        # 1. User & Item Encoders
        all_u = tf.expand_dims(self.user_embeddings, 1)
        if self.user_feature_mat is not None:
            all_u = tf.concat([all_u, tf.gather(self.user_feature_embeddings, self.user_feature_mat)], axis=1)

        all_i = tf.concat([tf.expand_dims(self.item_embeddings[:self.num_valid_items], 1),
                           tf.gather(self.item_feature_embeddings, self.item_feature_mat)], axis=1)

        encoded_u = tf.reduce_mean(all_u, axis=1) if self.num_gcn_layers > 0 else tf.reduce_sum(all_u, axis=1)
        encoded_i = tf.reduce_mean(all_i, axis=1) if self.num_gcn_layers > 0 else tf.reduce_sum(all_i, axis=1)

        # 2. Context Logic (The Fix for InvalidArgumentError)
        ctx_mat_batch = tf.gather(self.context_feature_mat, context_indices)

        ctx_wo_last = tf.gather(self.context_feature_embeddings, ctx_mat_batch[:, :-1])
        ctx_last = tf.expand_dims(tf.gather(self.item_embeddings, ctx_mat_batch[:, -1]), axis=1)

        batch_context_feature_embedding = tf.concat([ctx_wo_last, ctx_last],
                                                    axis=1)  # [batch, num_context_fields, hidden]

        #batch_context_feature_embedding = tf.squeeze(batch_context_feature_embedding, axis=1)

        # GNN Path
        insts_ctx_embed = tf.reduce_mean(tf.concat([
            tf.gather(self.context_feature_embeddings, self.context_feature_wo_last),
            tf.expand_dims(tf.gather(self.item_embeddings, self.context_feature_last), 1)
        ], axis=1), axis=1)

        insts_context_embedding = tf.gather(insts_ctx_embed, self.insts2contextid)

        layer_u, layer_i = encoded_u, encoded_i
        final_u_list = [layer_u * self.gcn_layer_weight[0]]
        final_i_list = [layer_i * self.gcn_layer_weight[0]]

        for k in range(self.num_gcn_layers):
            i_u = tf.gather(layer_u, self.insts2userid)
            i_i = tf.gather(layer_i, self.insts2itemid)

            if self.merge_type == 'sum':
                i_u_next = i_i + insts_context_embedding
                i_i_next = i_u + insts_context_embedding

            layer_u = tf.math.unsorted_segment_sum(i_u_next, self.insts2userid, self.num_users)
            layer_i = tf.math.unsorted_segment_sum(i_i_next, self.insts2itemid, self.num_valid_items)

            final_u_list.append(layer_u * self.gcn_layer_weight[k + 1])
            final_i_list.append(layer_i * self.gcn_layer_weight[k + 1])

        self.u_g_embeddings = tf.reduce_sum(final_u_list, axis=0)
        self.i_g_embeddings = tf.reduce_sum(final_i_list, axis=0)

        # 3. Decoder
        b_u_emb = tf.gather(self.u_g_embeddings, user_indices)
        b_i_emb = tf.gather(self.i_g_embeddings, item_indices)
        b_bias = tf.gather(self.user_bias, user_indices) + tf.gather(self.item_bias, item_indices)

        u_emb_dim = tf.expand_dims(b_u_emb, 1)
        i_emb_dim = tf.expand_dims(b_i_emb, 1)

        # Concatenate along axis 1 (the 'fields' axis)
        # Result shape: [batch_size, 1 + 1 + num_context_fields, hidden_factor]
        b_comb = tf.concat([u_emb_dim, i_emb_dim, batch_context_feature_embedding], axis=1)





        if self.decoder_type in ['FM', 'FM-Pooling']:
            if self.decoder_type == 'FM-Pooling':
                ctx_pooled = tf.reduce_mean(batch_context_feature_embedding, axis=1, keepdims=True)
                b_comb = tf.concat([u_emb_dim, i_emb_dim, ctx_pooled], axis=1)

            sq_sum = tf.square(tf.reduce_sum(b_comb, 1))
            sum_sq = tf.reduce_sum(tf.square(b_comb), 1)
            interaction = 0.5 * tf.reduce_sum(sq_sum - sum_sq, axis=1, keepdims=True)
            output = interaction + b_bias + self.global_bias

        elif self.decoder_type == 'IP':
            inner_product = tf.reduce_sum(b_u_emb * b_i_emb, axis=1, keepdims=True)
            output = inner_product + b_bias + self.global_bias
        else:
            # Fallback for other types
            output = b_bias + self.global_bias

        return tf.reshape(output, [-1])

    @tf.function
    def train_step(self, bat_users, bat_items, bat_context, bat_labels):
        with tf.GradientTape() as tape:
            preds = self({'user_id': bat_users, 'item_id': bat_items, 'context_id': bat_context})
            log_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=bat_labels, logits=tf.squeeze(preds)))
            reg_loss = self.reg * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'bias' not in v.name])
            total_loss = log_loss + reg_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss, log_loss

    @tf.function
    def train_step_bpr(self, bat_users, bat_context, bat_items_pos, bat_items_neg):
        with tf.GradientTape() as tape:
            pos_preds = self({'user_id': bat_users, 'item_id': bat_items_pos, 'context_id': bat_context})
            neg_preds = self({'user_id': bat_users, 'item_id': bat_items_neg, 'context_id': bat_context})
            bpr_loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_preds - neg_preds))
            reg_loss = self.reg * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'bias' not in v.name])
            total_loss = bpr_loss + reg_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss

    def train_model(self):
        user_input_val, context_input_val, item_input_val, labels_val = DataGenerator._get_pointwise_all_data_context(
            self.dataset, self.num_negatives, phase='valid')
        data_iter_val = DataIterator(user_input_val, context_input_val, item_input_val, labels_val,
                                     batch_size=self.test_batch_size, shuffle=False)

        for epoch in range(1, self.num_epochs + 1):
            t_start = time()
            total_loss = 0.0
            if self.loss_type == 'bpr_loss':
                u, c, i_p, i_n = DataGenerator._get_pairwise_all_data_context(self.dataset)
                data_iter = DataIterator(u, c, i_p, i_n, batch_size=self.batch_size, shuffle=True)
                for b_u, b_c, b_ip, b_in in data_iter:
                    total_loss += self.train_step_bpr(tf.cast(b_u, tf.int32), tf.cast(b_c, tf.int32),
                                                      tf.cast(b_ip, tf.int32), tf.cast(b_in, tf.int32)).numpy()
            else:
                u, c, i, l = DataGenerator._get_pointwise_all_data_context(self.dataset, self.num_negatives)
                data_iter = DataIterator(u, c, i, l, batch_size=self.batch_size, shuffle=True)
                for b_u, b_c, b_i, b_l in data_iter:
                    loss, _ = self.train_step(tf.cast(b_u, tf.int32), tf.cast(b_i, tf.int32), tf.cast(b_c, tf.int32),
                                              tf.cast(b_l, tf.float32))
                    total_loss += loss.numpy()

            logger.info(f"[Epoch {epoch}] loss: {total_loss / len(u):.6f} time: {time() - t_start:.1f}s")
            if epoch % args.test_interval == 0:
                buf, flag = self.evaluate()
                if flag: self.checkpoint.save(self.save_file)
                logger.info(f"epoch {epoch}: {buf}")

    def predict(self, user_ids, context_ids):
        ratings = np.empty([len(user_ids), self.num_valid_items])
        for i, (u_id, c_id) in enumerate(zip(user_ids, context_ids)):
            c_input = tf.tile(tf.expand_dims(tf.reshape(tf.convert_to_tensor(c_id), [-1]), 0),
                              [self.num_valid_items, 1])
            preds = self({'user_id': tf.fill([self.num_valid_items], int(u_id)),
                          'item_id': tf.range(self.num_valid_items, dtype=tf.int32),
                          'context_id': c_input}, training=False)
            ratings[i, :] = tf.squeeze(preds).numpy()
        return ratings

    @timer
    def evaluate(self):
        res, buf = self.evaluator.evaluate4CARS(self)
        if self.best_result[0] + self.best_result[2] < res[0] + res[2]:
            self.best_result, flag = res, True
        else:
            flag = False
        return buf, flag

    def build_graph(self):
        self._create_variables()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate) if self.optimizer_type == 'Adam' else tf.keras.optimizers.Adagrad(
            learning_rate=self.learning_rate)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
    @tf.function  # This makes it fast!
    def train_step(self, bat_users, bat_items, bat_context, bat_labels):
        with tf.GradientTape() as tape:
            # 1. Trigger the 'call' method to get predictions
            # This is where 'self.output' comes from now
            predictions = self({
                'user_id': bat_users,
                'item_id': bat_items,
                'context_id': bat_context
            })

            # 2. Calculate Prediction Loss (Log Loss)
            # We squeeze predictions to match label shape
            log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(bat_labels, tf.float32),
                logits=tf.squeeze(predictions)
            ))

            # 3. Calculate Regularization Loss
            # In TF2, we usually regularize the trainable weights directly
            reg_loss = self.reg * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'bias' not in v.name])

            total_loss = log_loss + reg_loss

        # 4. Backpropagation
        gradients = tape.gradient(total_loss, self.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None]

        self.optimizer.apply_gradients(grads_and_vars)

        return total_loss, log_loss

    #---------- training process -------
    def train_model(self):
        logger.info(self.evaluator.metrics_info())
        # Initial evaluation (Epoch 0)
        buf, flag = self.evaluate()
        logger.info("epoch 0:\t%s" % buf)

        # Prepare Validation Data (Pointwise used for loss tracking)
        user_input_val, context_input_val, item_input_val, labels_val = DataGenerator._get_pointwise_all_data_context(
            self.dataset, self.num_negatives, phase='valid'
        )
        data_iter_val = DataIterator(user_input_val, context_input_val, item_input_val, labels_val,
                                     batch_size=self.test_batch_size, shuffle=False)

        stopping_step = 0
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0.0
            training_start_time = time()

            # --- 1. TRAINING PHASE ---
            if self.loss_type == 'bpr_loss':
                # Pairwise Data Generation
                user_input, context_input, item_input_pos, item_input_neg = DataGenerator._get_pairwise_all_data_context(
                    self.dataset)
                data_iter = DataIterator(user_input, context_input, item_input_pos, item_input_neg,
                                         batch_size=self.batch_size, shuffle=True)

                time1 = time()
                for bat_users, bat_context, bat_items_pos, bat_items_neg in data_iter:
                    # Execute BPR Train Step
                    loss = self.train_step_bpr(
                        tf.convert_to_tensor(bat_users, dtype=tf.int32),
                        tf.convert_to_tensor(bat_context, dtype=tf.int32),
                        tf.convert_to_tensor(bat_items_pos, dtype=tf.int32),
                        tf.convert_to_tensor(bat_items_neg, dtype=tf.int32)
                    )
                    total_loss += loss.numpy()
            else:
                # Pointwise Data Generation
                user_input, context_input, item_input, labels = DataGenerator._get_pointwise_all_data_context(
                    self.dataset, self.num_negatives)
                data_iter = DataIterator(user_input, context_input, item_input, labels,
                                         batch_size=self.batch_size, shuffle=True)

                time1 = time()
                for bat_users, bat_context, bat_items, bat_labels in data_iter:
                    # Execute Log Loss Train Step
                    loss, _ = self.train_step(
                        tf.convert_to_tensor(bat_users, dtype=tf.int32),
                        tf.convert_to_tensor(bat_items, dtype=tf.int32),
                        tf.convert_to_tensor(bat_context, dtype=tf.int32),
                        tf.convert_to_tensor(bat_labels, dtype=tf.float32)
                    )
                    total_loss += loss.numpy()

            logger.info("[iter %d : loss: %f, time: %.1f = %.1f + %.1f]" % (
                epoch,
                total_loss / len(user_input),
                time() - training_start_time,
                time1 - training_start_time,
                time() - time1))

            # --- 2. VALIDATION LOSS TRACKING ---
            total_loss_val = 0.0
            for bat_users, bat_context, bat_items, bat_labels in data_iter_val:
                # Direct call to model for inference
                predictions = self({
                    'user_id': tf.convert_to_tensor(bat_users, dtype=tf.int32),
                    'item_id': tf.convert_to_tensor(bat_items, dtype=tf.int32),
                    'context_id': tf.convert_to_tensor(bat_context, dtype=tf.int32)
                })
                v_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(bat_labels, tf.float32),
                    logits=tf.squeeze(predictions)
                ))
                total_loss_val += v_loss.numpy()

            logger.info("[Validation loss @ %d: %.4f]" % (epoch, total_loss_val / len(user_input_val)))

            # --- 3. EVALUATION & CHECKPOINTING ---
            if epoch % args.test_interval == 0:
                buf, flag = self.evaluate()
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    logger.info("Find a better model.")

                else:
                    stopping_step += 1
                    if stopping_step >= args.stop_cnt:
                        logger.info("Early stopping triggered at epoch: {}".format(epoch))
                        break

                if self.save_flag > 0:
                    save_path = self.ckpt_manager.save()
                    logger.info(f"Checkpoint saved at: {save_path}")

                logger.info("epoch %d:\t%s" % (epoch, buf))

        # Final Summary
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        logger.info("best_result@epoch %d:\n" % self.best_epoch + buf)

    @timer
    def evaluate(self):
        """
        Evaluates the model using the evaluator.
        Removed sess.run and manual assignment ops as TF2 handles weight tracking automatically.
        """
        flag = False

        # In TF2, 'self' is passed directly to the evaluator.
        # When the evaluator calls self.predict(), it will now trigger the TF2 logic we wrote.
        current_result, buf = self.evaluator.evaluate4CARS(self)

        # Check if this is the best result so far (based on the sum of specific metrics)
        # Typically index 0 and 2 correspond to metrics like Recall or NDCG
        if self.best_result[0] + self.best_result[2] < current_result[0] + current_result[2]:
            self.best_result = current_result
            flag = True

        return buf, flag

    def predict(self, user_ids, context_ids):
        ratings = np.empty([len(user_ids), self.num_valid_items])

        for i, (user_id, context_id) in enumerate(zip(user_ids, context_ids)):
            # 1. Force context_id to be at least a 1D vector
            # If context_id is [1, 2, 3], it stays [1, 2, 3]
            # If context_id is 5, it becomes [5]
            ctx_tensor = tf.reshape(tf.convert_to_tensor(context_id), [-1])

            # 2. Expand and Tile
            # Shape becomes [1, num_context_fields]
            ctx_expanded = tf.expand_dims(ctx_tensor, 0)

            # The 'multiples' length must match the rank of ctx_expanded (which is 2)
            bat_context = tf.tile(ctx_expanded, [self.num_valid_items, 1])

            # 3. Rest of the inputs
            bat_users = tf.fill([self.num_valid_items], user_id)
            bat_items = tf.range(self.num_valid_items, dtype=tf.int32)

            # 4. Model Call
            preds = self({
                'user_id': bat_users,
                'item_id': bat_items,
                'context_id': bat_context
            }, training=False)

            ratings[i, :] = tf.squeeze(preds).numpy()

        return ratings

if __name__ == '__main__':
    # configurations
    model_name = 'GCM'
    args = configs.parse_args()
    args = configs.post_process_for_config(args, model_name)
    model_str = 'l%d-%s-%s-%s-reg%.0e' % (
        args.num_gcn_layers, 
        args.merge_type,
        args.decoder_type,
        args.adj_norm_type,
        args.reg,
    )

    # Logging
    current_time = strftime("%Y%m%d%H%M%S", localtime())
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # wirte into file
    ensureDir('%s/logs/%s-%s/' % (args.proj_path, model_name, args.dataset))
    fh = logging.FileHandler('%s/logs/%s-%s/%s_%s.log' % (args.proj_path, model_name, args.dataset, model_str, current_time))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # show on screen
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    # add two Hander
    logger.addHandler(fh)
    logger.addHandler(sh)

    # load data
    data = DATA(args, logger)

    Hyperparameters = ['model_name', 'dataset', 'epoch', 'batch_size', 'test_batch_size', 'test_interval', 'stop_cnt', 'decoder_type', 'hidden_factor', 'loss_type', 'lr', 'optimizer', 'pretrain', 'save_flag', 'reg', 'init_method', 'stddev', 'topk', 'num_gcn_layers', 'gcn_layer_weight', 'merge_type', 'adj_norm_type']
    if args.loss_type == 'log_loss':
        Hyperparameters.append('num_negatives')
    if args.decoder_type == 'MLP':
        Hyperparameters.append('num_hidden_layers')

    hyper_info = '\n'.join(["{}={}".format(arg, value) for arg, value in vars(args).items() if arg in Hyperparameters])
    logger.info('HyperParamters:\n' + hyper_info)
    if args.pretrain:
        ensureDir(args.proj_path + 'pretrain/pretrain-FM-%s/' % (args.dataset))
        # args.read_file = args.proj_path + 'pretrain/pretrain-FM-%s/embed_size=%d/' % (args.dataset, args.hidden_factor)
        args.read_file = "/home/btcchl0040/Documents/git-contribution/gcm-btc/pretrain/pretrain-GCM-Amazon-Book/l2-sum-FM-ls-reg1e-03/model"
    if args.save_flag:
        ensureDir(args.proj_path + 'pretrain/pretrain-GCM-%s/%s/' % (args.dataset, model_str))
        args.save_file = args.proj_path + 'pretrain/pretrain-GCM-%s/%s/model' % (args.dataset, model_str)


    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        # Model initialize
    model = GCM(data, args)
    model.build_graph()
    model._create_variables()

    checkpoint_dir = os.path.join(args.proj_path, 'checkpoints', model_name, args.dataset)
    ensureDir(checkpoint_dir)
    model.ckpt_manager = tf.train.CheckpointManager(
        model.checkpoint,
        directory=checkpoint_dir,
        max_to_keep=5  # keeps last 5 checkpoints
    )
    if model.ckpt_manager.latest_checkpoint:
        model.checkpoint.restore(model.ckpt_manager.latest_checkpoint)
        logger.info(f"Resumed from checkpoint: {model.ckpt_manager.latest_checkpoint}")
    else:
        logger.info("No checkpoint found, training from scratch.")

    # Training
    logger.info(
        '########################### begin training ###########################'
    )
    model.train_model()
    logger.info(
        '########################### end training ###########################')

    # Prediction
    logger.info("Prediction flow:")
    user_id = input('Enter User id : ')
    context_data = input('Enter Context data : ')
    print(model.predict([user_id], [context_data]))
