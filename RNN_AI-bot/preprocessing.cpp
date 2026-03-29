#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
using namespace std;
using ll = long long;
using vec = vector<int>;
using itr = istream_iterator<int>;
using otr = ostream_iterator<int>;
int main() {
	string temp;
	ifstream is("dialogues_text.txt");
	if (is.fail()) {
		cout << 1 << endl; return 1;
	}
	unordered_map<string, int> mp;
	while (is >> temp) {
		mp[temp]++;
	}
	multimap<int, string,greater<int>> mm;
	for (auto [word, num] : mp) {
		mm.emplace(num, word);
	}
	int iter = 1;
	unordered_map<string, int> cc;
	ofstream os("word_list.txt");
	for (auto [num, word] : mm) {
		os << word << ' ' << iter << '\n';
		cc[word] = iter++;
	}
	os.close(); is.clear(); is.seekg(0);
	ofstream oo("text_index.txt");
	string a;
	while (getline(is, temp)) {
		stringstream ss(temp);
		while (ss >> a) oo << cc[a] << ' ';
		oo << 0 << ' ';//0 is \n
	}
	is.close(); oo.close();
}