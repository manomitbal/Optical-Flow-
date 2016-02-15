#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <unistd.h>
#define MAXHOSTNAME 256
using namespace std;

main()
{
   struct sockaddr_in socketInfo;
   char sysHost[MAXHOSTNAME+1];  // Hostname of this computer we are running on
   struct hostent *hPtr;
   int socketHandle;
   int portNumber = 1989;

   bzero(&socketInfo, sizeof(sockaddr_in));  // Clear structure memory

   // Get system information

   gethostname(sysHost, MAXHOSTNAME);  // Get the name of this computer we are running on
   if((hPtr = gethostbyname(sysHost)) == NULL)
   {
      cerr << "System hostname misconfigured." << endl;
      exit(EXIT_FAILURE);
   }

   // create socket

   if((socketHandle = socket(AF_INET, SOCK_STREAM, 0)) < 0)
   {
      close(socketHandle);
      exit(EXIT_FAILURE);
   }

   // Load system information into socket data structures

   socketInfo.sin_family = AF_INET;
   socketInfo.sin_addr.s_addr = htonl(INADDR_ANY); // Use any address available to the system
   socketInfo.sin_port = htons(portNumber);      // Set port number

   // Bind the socket to a local socket address

   if( bind(socketHandle, (struct sockaddr *) &socketInfo, sizeof(socketInfo)) < 0)
   {
      close(socketHandle);
      perror("bind");
      exit(EXIT_FAILURE);
   }
   while(1)
   {
      listen(socketHandle, 1);

      int socketConnection;
      if( (socketConnection = accept(socketHandle, NULL, NULL)) < 0)
      {
         exit(EXIT_FAILURE);
      }
      

      int rc = 0;  // Actual number of bytes read
      char buf[512];

   // rc is the number of characters returned.
   // Note this is not typical. Typically one would only specify the number 
   // of bytes to read a fixed header which would include the number of bytes
   // to read. See "Tips and Best Practices" below.

      rc = recv(socketConnection, buf, 512, 0);
      buf[rc]= (char) NULL;        // Null terminate string
      cout << buf << endl;
   }
}
          