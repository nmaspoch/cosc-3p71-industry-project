# COSC 3P71 Industry Project

### Start label-studio

If you do not have label-studio installed, run the following command:

```
$ pip install label-studio
```

Go into mapillary_images folder, create a terminal and start CORS Server

```
$ cd mapillary_images
$ python3 ../cors_server.py
```

In the root folder, create another terminal and run the following command

```
$ label-studio start
```

A new window will be created at localhost:8080
