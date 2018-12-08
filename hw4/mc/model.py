import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        dd = d#d/2
        dropout = 0.5  # no dropout = 1
        with tf.variable_scope('GRU_X'):
            if config.is_train:
                fwcellx = DropoutWrapper(GRUCell(dd), dropout)
                bwcellx = DropoutWrapper(GRUCell(dd), dropout)

                #for the 2 layer
                fwcellx2 = DropoutWrapper(GRUCell(dd), dropout)
                bwcellx2 = DropoutWrapper(GRUCell(dd), dropout)
            else:
                fwcellx = GRUCell(dd)
                bwcellx = GRUCell(dd)

                #for the 2 layer
                fwcellx2 = GRUCell(dd)
                bwcellx2 = GRUCell(dd)

            with tf.variable_scope('layer1'):
                outputx = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellx, cell_bw=bwcellx, inputs=xx, sequence_length=x_len, dtype=tf.float32)
                (outputsx, output_statesx) = outputx
                concatx = tf.concat(outputsx, 2)
            with tf.variable_scope('layer2'):
                outputx = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellx2, cell_bw=bwcellx2, inputs=concatx, dtype=tf.float32)
                (outputsx, output_statesx) = outputx
                concatx = tf.concat(outputsx, 2)
            print(concatx)

        with tf.variable_scope('GRU_Q'):
            if config.is_train:
                fwcellq = DropoutWrapper(GRUCell(dd), dropout)
                bwcellq = DropoutWrapper(GRUCell(dd), dropout)

                fwcellq2 = DropoutWrapper(GRUCell(dd), dropout)
                bwcellq2 = DropoutWrapper(GRUCell(dd), dropout)
            else:
                fwcellq = GRUCell(dd)
                bwcellq = GRUCell(dd)

                fwcellq2 = GRUCell(dd)
                bwcellq2 = GRUCell(dd)

            with tf.variable_scope('layer1'):
                outputq = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellq, cell_bw=bwcellq,
                                                      inputs=qq, sequence_length=q_len,
                                                      dtype=tf.float32)
                (outputsq, output_statesq) = outputq
                concatq = tf.concat(outputsq, 2)
            with tf.variable_scope('layer2'):
                outputq = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellq2, cell_bw=bwcellq2,
                                                      inputs=concatq,dtype=tf.float32)
                (outputsq, output_statesq) = outputq
                concatq = tf.concat(outputsq, 2)
        concatq = tf.layers.dense(concatq,d)
        concatx = tf.layers.dense(concatx,d)
        xx = concatx
        qq = concatq

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        dd = d  # d/2
        dropout = 0.5  # no dropout = 1
        with tf.variable_scope('GRU_X'):
            if config.is_train:
                fwcellx = DropoutWrapper(GRUCell(dd), dropout)
                bwcellx = DropoutWrapper(GRUCell(dd), dropout)

                # for the 2 layer
                fwcellx2 = DropoutWrapper(GRUCell(dd), dropout)
                bwcellx2 = DropoutWrapper(GRUCell(dd), dropout)
            else:
                fwcellx = GRUCell(dd)
                bwcellx = GRUCell(dd)

                # for the 2 layer
                fwcellx2 = GRUCell(dd)
                bwcellx2 = GRUCell(dd)

            with tf.variable_scope('layer1'):
                outputx = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellx, cell_bw=bwcellx, inputs=xx,
                                                          sequence_length=x_len, dtype=tf.float32)
                (outputsx, output_statesx) = outputx
                concatx = tf.concat(outputsx, 2)
            with tf.variable_scope('layer2'):
                outputx = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellx2, cell_bw=bwcellx2, inputs=concatx,
                                                          dtype=tf.float32)
                (outputsx, output_statesx) = outputx
                concatx = tf.concat(outputsx, 2)
            print(concatx)

        with tf.variable_scope('GRU_Q'):
            if config.is_train:
                fwcellq = DropoutWrapper(GRUCell(dd), dropout)
                bwcellq = DropoutWrapper(GRUCell(dd), dropout)

                fwcellq2 = DropoutWrapper(GRUCell(dd), dropout)
                bwcellq2 = DropoutWrapper(GRUCell(dd), dropout)
            else:
                fwcellq = GRUCell(dd)
                bwcellq = GRUCell(dd)

                fwcellq2 = GRUCell(dd)
                bwcellq2 = GRUCell(dd)

            with tf.variable_scope('layer1'):
                outputq = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellq, cell_bw=bwcellq,
                                                          inputs=qq, sequence_length=q_len,
                                                          dtype=tf.float32)
                (outputsq, output_statesq) = outputq
                concatq = tf.concat(outputsq, 2)
            with tf.variable_scope('layer2'):
                outputq = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwcellq2, cell_bw=bwcellq2,
                                                          inputs=concatq, dtype=tf.float32)
                (outputsq, output_statesq) = outputq
                concatq = tf.concat(outputsq, 2)
        concatq = tf.layers.dense(concatq, d)
        concatx = tf.layers.dense(concatx, d)
        
        xx = concatx
        qq = concatq

        print("xx")
        print(xx)

        print("qq")
        print(qq)

        xx_qq_xxqq = concatMatrix(xx, qq, JX, JQ, d, 3)#[N,JX,JQ,3d]
        print("xx_qq_xxqq")
        print(xx_qq_xxqq)

        xx_qq_xxqq_flat = tf.reshape(xx_qq_xxqq, [-1, 3*d]) #[N*JX*JQ,3d]
        print("xx_qq_xxqq_flat")
        print(xx_qq_xxqq_flat)

        xx_qq_xxqq = tf.reshape(tf.layers.dense(inputs=xx_qq_xxqq_flat, units=1), [-1, JX, JQ]) #[N,JX,JQ]
        print("dense xx_qq_xxqq")
        print(xx_qq_xxqq)

        masked_matrix = mask_dim(xx_qq_xxqq, q_mask, 1)
        pk = tf.nn.softmax(masked_matrix, axis=2)
        print("pk")
        print(pk)

        qkb = tf.matmul(pk, qq)
        print("qkb mult")
        print(qkb)

        xq = tf.concat([xx, qkb, xx * qkb], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def concatMatrix(xx, qq, jx, jq, d, d_rd=3):
    xxx = tf.expand_dims(xx, axis=2)
    xxx = tf.tile(xxx, [1, 1, jq, 1])
    qqq = tf.expand_dims(qq, axis=1)
    qqq = tf.tile(qqq, [1, jx, 1, 1])
    xxxqqqq = xxx * qqq
    output = tf.concat([xxx, qqq, xxxqqqq], axis=3)
    return output

def mask_dim(matrix,mask,dim):
    mask = tf.expand_dims(mask, dim)
    masked_matrix = matrix - (1.0 - tf.cast(mask,'float')) * 10.0e10
    return masked_matrix

def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
