from appCloud import app
import config

"""
iNITIALIZE Cloud API
"""

# configuring Host and Port from configuration files. 
app.debug = False
app.run(host=config.HOST_CLOUD, port=config.PORT_CLOUD)
