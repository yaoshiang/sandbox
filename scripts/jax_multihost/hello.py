import jax

jax.distributed.initialize()
print(f"Host {jax.process_index()} checked in! Local TPUs: {len(jax.local_devices())} Global TPUs: {jax.device_count()}")
