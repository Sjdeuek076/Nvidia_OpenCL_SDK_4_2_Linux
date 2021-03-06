#ifndef pstream_h
#define pstream_h
//-------------------------------------------------------------------------------------------
#include <cstdio>
#include <iostream>
#include <sstream>
//-------------------------------------------------------------------------------------------
namespace hogehoge
{
//-------------------------------------------------------------------------------------------
namespace ps
{
//-------------------------------------------------------------------------------------------
struct __pendl {char dummy;} endl;
struct __pflush {char dummy;} flush;

class pipestream
{
private:
  FILE  *p;
public:
  pipestream (void) : p(NULL) {};
  pipestream (const char *name) : p(NULL) {open(name);};
  ~pipestream (void) {close();}
  bool open (const char *name)
    {
      if (is_open())  close();
      p = popen(name, "w");
      if (p == NULL)
      {
        std::cerr<<"failed to open "<<name<<std::endl;
        return false;
      }
      return true;
    };
  void close (void)
    {
      if(is_open())  {pclose(p); p=NULL;}
    };
  bool is_open (void) const {return p!=NULL;};
  template <typename T>
  pipestream& operator<< (const T &rhs)
    {
      std::stringstream ss;
      ss<<rhs;
      fputs(ss.str().c_str(), p);
      return *this;
    };
  pipestream& operator<< (const char *rhs)
    {
      fputs(rhs, p);
      return *this;
    };
  pipestream& operator<< (const __pendl &rhs)
    {
      fputs("\n", p);
      fflush(p);
      return *this;
    };
  pipestream& operator<< (const __pflush &rhs)
    {
      fflush(p);
      return *this;
    };
};
//-------------------------------------------------------------------------------------------
}  // end of namespace ps
//-------------------------------------------------------------------------------------------
}  // end of namespace hogehoge
//-------------------------------------------------------------------------------------------
#endif // pstream_h
