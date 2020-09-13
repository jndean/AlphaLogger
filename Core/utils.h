

#define MALLOC_CHECK(x) \
    do {if ((x) == NULL) { \
        fprintf(stderr, "Malloc failure: function \"%s\" in %s:%d\n", \
	        (char*)__func__, (char*)__FILE__, __LINE__); \
	    exit(1701); \
    }} while (0)
