from threading import Thread, RLock
from functools import wraps, update_wrapper, _CacheInfo
from typing import Optional, Callable, Any, TypeVar, Generic
import json

OUT = TypeVar('OUT')
THN = TypeVar('THN')
EXC = TypeVar('EXC')


class Promise(Thread, Generic[OUT]):
    ''' A thread-based Promise. To be used together with promise function/decorator.
        Promised function will be executed in a separate thread and the result will be accessible
        via the result property. If the function raises and exception, calling the result property
        will raise the same exception. The exception property can be used to check if there is an exception
        as it either returns a raised exception or None.
    '''

    def __init__(self,
                 func: Callable[[], OUT],
                 *,
                 group: Optional[Any] = None,
                 name: Optional[str] = None,
                 daemon: Optional[bool] = None):
        ''' Create a promise object that will execute the given function in a separate thread.
            The result of the function will be accessible via the result property.
            If the function raises an exception, the exception will be accessible via the exception property.
            Builder pattern can be used to chain functions to be executed after this promise is done.
            The function will be passed the result of this promise as argument.

            Args:
                func (Callable[[], OUT]): The function to be executed in a separate thread.
            Kwargs (optional, see threading.Thread):
                group (Optional[Any], optional): Thread group. Defaults to None.
                name (Optional[str], optional): Thread name. Defaults to None.
                daemon (Optional[bool], optional): Daemon flag. Defaults to None.
        '''
        super().__init__(group=group, name=name, daemon=daemon)
        self._func: Callable = func
        self._result: Optional[OUT] = None
        self._exception: Optional[Exception] = None
        self.start()

    def __repr__(self) -> str:
        ''' Return a string representation of the promise object
        '''
        return f'<Promise({self._func.__name__})>'

    def run(self) -> None:
        ''' Run the promise function and set the result. Capture exception if it is raised
        '''
        try:
            self._result = self._func()
        except Exception as e:
            self._exception = e

    def then(self, func: Callable[[OUT], THN]) -> 'Promise[THN]':
        ''' Chain a function to be executed after this promise is done.
            The function will be passed the result of this promise as argument.
        '''
        @wraps(func)
        def wrapper():
            return func(self.result)
        return Promise(wrapper)

    def catch(self, func: Callable[[Exception], EXC]) -> 'Promise[EXC]':
        ''' Chain a function to be executed if this promise raises an exception.
            The function will be passed the exception as argument.
        '''
        @wraps(func)
        def wrapper():
            if self.exception:
                return func(self.exception)
        return Promise(wrapper)

    @property
    def result(self) -> OUT:
        ''' Get the result of the function. This will cause the thread to join, so it will block until the function is done.
            If the function raised an exception, it will be raised here.
        '''
        self.join()
        if self._exception:
            raise self._exception
        return self._result  # type: ignore

    @property
    def exception(self) -> Optional[Exception]:
        ''' Get the exception raised by the function or return None if no exception.
            This will cause the thread to join, so it will block until the function is done.
        '''
        self.join()
        return self._exception

    def result_with_timeout(self, timeout: float) -> OUT:
        ''' Get the result of the function. This will cause the thread to join, so it will block until the function is done.
            If the function raised an exception, it will be raised here.
            If the function takes longer than timeout, TimeoutError will be raised.

            Args:
                timeout (float): Time in seconds to wait for the function to complete. If not completed within the timeout, TimeoutError is raised.
        '''

        self.join(timeout=timeout)
        if self.is_alive():
            self._exception = TimeoutError(
                f'{self} timed out after {timeout} seconds')

        if self._exception:
            raise self._exception
        return self._result  # type: ignore


def promise(func: Callable[..., OUT]) -> Callable[..., Promise[OUT]]:
    ''' Decorator for a function that will be executed in a separate thread.
        The property "result" will be set to the result of the function when it is done.
        Getting the "result" will cause the thread to join, so it will block until the function is done.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs) -> Promise[OUT]:

        @wraps(func)
        def closured_call():
            ''' This allows the promise object to return as <Promise(func_name)> even though
                what is passed to Promise is a lambda function that calls the original function
                with arguments passed via closure.
            '''
            return func(*args, **kwargs)
        return Promise(closured_call)

    return wrapper


def _make_key(args: list, kwargs: dict):
    def to_dict(arg):
        if hasattr(arg, 'dict') and callable(arg.dict):
            return arg.dict()
        else:
            return arg

    return hash(json.dumps([to_dict(arg) for arg in args])
                + json.dumps({k: to_dict(v) for k, v in kwargs.items()}))


def json_cache(maxsize=128):
    """Least-recently-used cache decorator. LRU cache that hashes input using json string.
    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.
    See:  https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)
    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    if isinstance(maxsize, int):
        # Negative maxsize is treated as 0
        if maxsize < 0:
            maxsize = 0
    elif callable(maxsize):
        # The user_function was passed in directly via the maxsize argument
        user_function, maxsize = maxsize, 128
        wrapper = _json_cache_wrapper(user_function, maxsize, _CacheInfo)
        wrapper.cache_parameters = lambda: {'maxsize': maxsize}
        return update_wrapper(wrapper, user_function)
    elif maxsize is not None:
        raise TypeError(
            'Expected first argument to be an integer, a callable, or None')

    def decorating_function(user_function):
        wrapper = _json_cache_wrapper(user_function, maxsize, _CacheInfo)
        wrapper.cache_parameters = lambda: {'maxsize': maxsize}
        return update_wrapper(wrapper, user_function)

    return decorating_function


def _json_cache_wrapper(user_function, maxsize, _CacheInfo):
    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = _make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3   # names for the link fields

    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get    # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()           # because linkedlist updates aren't threadsafe
    root = []                # root of the circular doubly linked list
    root[:] = [root, root, None, None]     # initialize by pointing to self

    if maxsize == 0:

        def wrapper(*args, **kwds):
            # No caching -- just a statistics update
            nonlocal misses
            misses += 1
            result = user_function(*args, **kwds)
            return result

    elif maxsize is None:

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        def wrapper(*args, **kwds):
            # Size limited caching that tracks accesses by recency
            nonlocal root, hits, misses, full
            key = make_key(args, kwds)
            with lock:
                link = cache_get(key)
                if link is not None:
                    # Move the link to the front of the circular queue
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = user_function(*args, **kwds)
            with lock:
                if key in cache:
                    # Getting here means that this same key was added to the
                    # cache while the lock was released.  Since the link
                    # update is already done, we need only return the
                    # computed result and update the count of misses.
                    pass
                elif full:
                    # Use the old root to store the new key and result.
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    # Empty the oldest link and make it the new root.
                    # Keep a reference to the old key and old result to
                    # prevent their ref counts from going to zero during the
                    # update. That will prevent potentially arbitrary object
                    # clean-up code (i.e. __del__) from running while we're
                    # still adjusting the links.
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    # oldresult = root[RESULT]
                    root[KEY] = root[RESULT] = None
                    # Now update the cache dictionary.
                    del cache[oldkey]
                    # Save the potentially reentrant cache[key] assignment
                    # for last, after the root and links have been put in
                    # a consistent state.
                    cache[key] = oldroot
                else:
                    # Put result in a new link at the front of the queue.
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    # Use the cache_len bound method instead of the len() function
                    # which could potentially be wrapped in an lru_cache itself.
                    full = (cache_len() >= maxsize)
            return result

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper