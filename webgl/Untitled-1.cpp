#include <iostream>
#include <list>
#include <string>

class IObserver
{
public:
    virtual ~IObserver(){};
    virtual void Update(const std::string message_from_subject) = 0;
}

class ISubject
{
public:
    virtual ~ISubject(){};
    virtual void Attach(IObserver *observer) = 0;
    virtual void Detach(IObserver *observer) = 0;
    virtual void Notify() = 0;
}

class Subject : public ISubject
{
    std::list<IObserver *> list_observer_;
    std::string message_;

public:
    virtual ~Subject()
    {
        std::cout << "Goodbye,Subject \n";
    }
    void Attach(IObserver *observer) override
    {
        list_observer_.push_back(observer);
    }
    void Detach(Iobserver *observer) override
    {
        list_observer_.push_back(object);
    }
    void Notify() override
    {
        std::list<IObserver *>::iterator iterator = list_observer_.begin();
        std::cout << list_observer_.size() << "observer in the list \n";
        while (iterator != list_observer_.end())
        {
            (*iterator)->Update(message_);
            ++iterator;
        }
    }
    void CreateMessage(std::string message = "Empty")
    {
        this->message_ = message;
        Notify();
    }
    void SameNotifyFunction()
    {
        this->message_ = "same message";
        Notify();
    }
    class Observer : IObserver
    {
    private:
        std::string message_from_subject_;
        Subject &subject;
        Observer(Subject &subject) : subject_(subject)
        {
            this->subject_.Attach(this);
            std::cout << "Create observer" << '\n';
        }
        virtual ~Observer()
        {
        std:
            cout << "Delete observer" << '\n';
        }
        void Update(const std::string &message_from_object) override
        {
            message_from_object_ = message_from_object;
            std::cout << "meesage: " << message_from_object_ << '\n';
        }
        void RemoveFromList()
        {
            subject_.Detach(this);
        }
    }
} class Fish
{
public:
    void virtual swim() = 0;
};
class MString
{
private:
    char *buffer;

public:
    MString(char *str)
    {
        *buffer = new char[strlen(str) + 1];
        strcpy(buffer, str);
        std::cout << "init MString" << std::endl;
    }
    MString(const MString &CopyString)
    {
        if (CopyString.buffer != NULL)
        {
            buffer = new char[strlen(CopyString.buffer) + 1];
            strcpy(buffer, CopyString.buffer);
        }
    }
    ~MString()
    {
        delete buffer;
    }
    int GetLength()
    {
        return strlen(buffer);
    }
    char *GetString()
    {
        return buffer;
    }
}
class Date{
    private:
        int day;
        int month;
        int year;
        std::string DayInStr;
    public:
        Date(int inDay,int inMonth,int inYear):day(inDay),month(inMonth),year(inYear){};
        std::string getDays()const{
            return "0";
        }
        operator const*(){
            ostringstream formattedDate;
            formattedDate << day << "/" << month << "/" << year << std::endl;
            DatelnString = formattedDate.str();
            return DatelnString.c_str();
        }

};
class Point{
    public:
    Point(int x,y):X(x),Y(y){};
    Point(const Point& p){X = p.X;Y=p.Y};
    private: int X,Y;
};
struct s{
    int x;
    int y;
};
typedef struct s t1;
typedef double (*MATH)();
MATH cos;