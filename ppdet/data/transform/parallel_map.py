# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function:
#   transform samples in 'source' using 'mapper'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
if six.PY3:
    from queue import Empty
else:
    from Queue import Empty

import uuid
import logging
import signal
import threading
from .transformer import ProxiedDataset

logger = logging.getLogger(__name__)


class EndSignal(object):
    """ signal used to notify worker to exit
    """
    def __init__(self, id, errno=0, errmsg=''):
        self.id = id
        self.errno = errno
        self.errmsg = errmsg


class ParallelMappedDataset(ProxiedDataset):
    """
    Transform samples to mapped samples which is similar to 
    'basic.MappedDataset', but multiple workers (threads or processes) 
    will be used

    Notes:
        this class is not thread-safe
    """

    def __init__(self, source, mapper, worker_args):
        super(ParallelMappedDataset, self).__init__(source)
        worker_args = {k.lower(): v for k, v in worker_args.items()}

        args = {
            'bufsize': 100,
            'worker_num': 8,
            'use_process': False,
            'memsize': '3G'
        }
        args.update(worker_args)
        if args['use_process'] and type(args['memsize']) is str:
            assert args['memsize'][-1].lower() == 'g', \
                "invalid param for memsize[{}], should " \
                "be ended with 'G' or 'g'".format(args['memsize'])
            gb = args['memsize'][:-1]
            args['memsize'] = int(gb) * 1024 ** 3

        self._worker_args = args
        self._started = False
        self._source = source
        self._mapper = mapper
        self._exit = False
        self._setup()

    def _setup(self):
        """setup input/output queues and workers """
        use_process = self._worker_args.get('use_process', False)
        if use_process and sys.platform == "win32":
            logger.info("Use multi-thread reader instead of "
                        "multi-process reader on Windows.")
            use_process = False

        bufsize = self._worker_args['bufsize']
        if use_process:
            from .shared_queue import SharedQueue as Queue
            from multiprocessing import Process as Worker
            from multiprocessing import Event
            memsize = self._worker_args['memsize']
            self._inq = Queue(bufsize, memsize=memsize)
            self._outq = Queue(bufsize, memsize=memsize)
        else:
            if six.PY3:
                from queue import Queue
            else:
                from Queue import Queue
            from threading import Thread as Worker
            from threading import Event
            self._inq = Queue(bufsize)
            self._outq = Queue(bufsize)

        consumer_num = self._worker_args['worker_num']
        id = str(uuid.uuid4())[-3:]
        self._producer = threading.Thread(
            target=self._produce,
            args=('producer-' + id, self._source, self._inq))
        self._producer.daemon = True

        self._consumers = []
        self._consumer_endsig = {}
        for i in range(consumer_num):
            consumer_id = 'consumer-' + id + '-' + str(i)
            p = Worker(
                target=self._consume,
                args=(consumer_id, self._inq, self._outq,
                      self._mapper))
            self._consumers.append(p)
            p.daemon = True
            setattr(p, 'id', consumer_id)

        self._epoch = -1
        self._feeding_ev = Event()
        self._produced = 0  # produced sample in self._produce
        self._consumed = 0  # consumed sample in self.next

    def _produce(self, id, source, inq):
        """Fetch data from source and feed it to 'inq' queue"""
        endsig = EndSignal(id)
        while True:
            self._feeding_ev.wait()
            if self._exit:
                break
            try:
                inq.put(source.next())
                self._produced += 1
            except StopIteration:
                self._feeding_ev.clear()
                self._feeding_ev.wait()
            except Exception as e:
                endsig.errno = -1
                endsig.errmsg = "producer[{}] failed with error: {}" \
                    .format(id, str(e))
                inq.put(endsig)
                break

    def _consume(self, id, inq, outq, mapper):
        """Fetch data from 'inq', process it and put result to 'outq'"""
        if self._worker_args['use_process']:
            # handle SIGTERM signal to exit to prevent print stack frame
            signal.signal(signal.SIGTERM, lambda signum, frame : sys.exit())

        endsig = EndSignal(id)
        while True:
            sample = inq.get()
            if isinstance(sample, EndSignal):
                endsig.errno = sample.errno
                endsig.errmsg = "consumer[{}] exits for reason[{}]" \
                    .format(id, sample.errmsg)
                outq.put(endsig)
                break

            try:
                result = mapper(sample)
                outq.put(result)
            except Exception as e:
                endsig.errno = -2
                endsig.errmsg = "consumer[{}] failed to map with error:[{}]" \
                    .format(id, str(e))
                outq.put(endsig)
                break

    def drained(self):
        assert self._epoch >= 0, "first epoch has not started yet"
        return self._source.drained() and self._produced == self._consumed

    def stop(self):
        """ notify to exit
        """
        self._exit = True
        self._feeding_ev.set()
        for _ in range(len(self._consumers)):
            self._inq.put(EndSignal(0, "notify consumers to exit"))

    def _consumer_healthy(self):
        abnormal_num = 0
        for w in self._consumers:
            if not w.is_alive() and w.id not in self._consumer_endsig:
                abnormal_num += 1
                if self._worker_args['use_process']:
                    errmsg = "consumer[{}] exit abnormally with exitcode[{}]" \
                                .format(w.pid, w.exitcode)
                else:
                    errmsg = "consumer[{}] exit abnormally".format(w.ident)
    
                logger.warn(errmsg)

        if abnormal_num > 0:
            logger.warn("{} consumers have exited abnormally!!!" \
                .format(abnormal_num))

        return abnormal_num == 0

    def next(self):
        """ get next transformed sample
        """
        if self._epoch < 0:
            self.reset()

        if self.drained():
            raise StopIteration()

        while not self._exit:
            try:
                sample = self._outq.get(timeout=3)
            except Empty as e:
                if not self._consumer_healthy():
                    raise StopIteration()
                else:
                    continue

            if isinstance(sample, EndSignal):
                self._consumer_endsig[sample.id] = sample
                logger.warn("recv endsignal from outq with errmsg[{}]" \
                    .format(sample.errmsg))

                if len(self._consumer_endsig.keys()) < len(self._consumers):
                    self._inq.put(sample)
                else:
                    self._exit = True
                    raise StopIteration("all consumers exited, no more samples")
            else:
                self._consumed += 1
                return sample

        raise StopIteration()

    def reset(self):
        """ reset for a new epoch of samples
        """
        assert not self._exit, "cannot reset for already stopped dataset"

        if self._epoch < 0:
            self._epoch = 0
            for w in self._consumers:
                w.start()
            self._producer.start()
        else:
            assert self._consumer_healthy(), "cannot start another pass of data" \
                " for some consumers exited abnormally before!!!"

            if not self.drained():
                logger.warn("reset before epoch[{}] finishes".format(self._epoch))
                self._produced = self._produced - self._consumed
            else:
                self._produced = 0

            self._epoch += 1

        assert len(self._consumer_endsig.keys()) == 0, "some consumers already exited," \
            + " cannot start another epoch"

        self._source.reset()
        self._consumed = 0
        self._feeding_ev.set()


# FIXME(dengkaipeng): fix me if you have better impliment
# handle terminate reader process, do not print stack frame
signal.signal(signal.SIGTERM, lambda signum, frame : sys.exit())
