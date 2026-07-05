"""Local MeshCat subprocess compatibility patch for Windows SSL stores."""

import ssl

ssl.create_default_context = lambda *args, **kwargs: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
